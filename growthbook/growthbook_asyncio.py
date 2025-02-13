import inspect
import json
from abc import ABC, abstractmethod
from time import time
from typing import Awaitable, Optional, Dict, List, Any, Set

import aiohttp
from urllib3 import PoolManager

from . import AgnosticGrowthBookBase
from .common_types import (
    AbstractAsyncStickyBucketService,
    AsyncGlobalContext,
    AsyncEvaluationContext,
    StackContext,
)
from .core import _getHashValue
from .core_async import run_experiment_async, eval_feature_async
from .growthbook import (
    CacheEntry,
    SSEClient,
    logger,
    decrypt,
    Feature,
    FeatureResult,
    Experiment,
    Result,
)


class AbstractAsyncFeatureCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Awaitable[Optional[Dict]]:
        pass

    @abstractmethod
    def set(self, key: str, value: Dict, ttl: int) -> Awaitable[None]:
        pass

    async def clear(self) -> None:
        pass


class AsyncInMemoryFeatureCache(AbstractAsyncFeatureCache):
    def __init__(self) -> None:
        self.cache: Dict[str, CacheEntry] = {}

    async def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            entry = self.cache[key]
            if entry.expires >= time():
                return entry.value
        return None

    async def set(self, key: str, value: Dict, ttl: int) -> None:
        if key in self.cache:
            self.cache[key].update(value)
        self.cache[key] = CacheEntry(value, ttl)

    async def clear(self) -> None:
        self.cache.clear()


class AsyncInMemoryStickyBucketService(AbstractAsyncStickyBucketService):
    def __init__(self) -> None:
        self.docs: Dict[str, Dict] = {}

    async def get_assignments(
        self, attributeName: str, attributeValue: str
    ) -> Optional[Dict]:
        return self.docs.get(self.get_key(attributeName, attributeValue), None)

    async def save_assignments(self, doc: Dict) -> None:
        self.docs[self.get_key(doc["attributeName"], doc["attributeValue"])] = doc

    async def destroy(self) -> None:
        self.docs.clear()


class AsyncFeatureRepository(object):
    def __init__(self) -> None:
        self.cache: AbstractAsyncFeatureCache = AsyncInMemoryFeatureCache()
        self.http: Optional[PoolManager] = None
        self.sse_client: Optional[SSEClient] = None

    def set_cache(self, cache: AbstractAsyncFeatureCache) -> None:
        self.cache = cache

    async def clear_cache(self):
        await self.cache.clear()

    async def save_in_cache(self, key: str, res, ttl: int = 60):
        await self.cache.set(key, res, ttl)

    async def load_features(
        self, api_host: str, client_key: str, decryption_key: str = "", ttl: int = 60
    ) -> Optional[Dict]:
        if not client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        key = api_host + "::" + client_key

        cached = await self.cache.get(key)
        if not cached:
            res = await self._fetch_features(api_host, client_key, decryption_key)
            if res is not None:
                await self.cache.set(key, res, ttl)
                logger.debug("Fetched features from API, stored in cache")
                return res
        return cached

    async def _get_json_or_none(self, url: str) -> Optional[Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status >= 400:
                        logger.warning(
                            "Failed to fetch features, received status code %d",
                            response.status,
                        )
                        return None
                    decoded = await response.json()
                    return decoded
        except aiohttp.ClientError as e:
            logger.warning(f"HTTP request failed: {e}")
            return None
        except Exception as e:
            logger.warning("Failed to decode feature JSON from GrowthBook API: %s", e)
            return None

    async def _fetch_and_decode(self, api_host: str, client_key: str) -> Optional[Dict]:
        url = self._get_features_url(api_host, client_key)
        return await self._get_json_or_none(url)

    def decrypt_response(self, data, decryption_key: str):
        if "encryptedFeatures" in data:
            if not decryption_key:
                raise ValueError("Must specify decryption_key")
            try:
                decryptedFeatures = decrypt(data["encryptedFeatures"], decryption_key)
                data["features"] = json.loads(decryptedFeatures)
                del data["encryptedFeatures"]
            except Exception:
                logger.warning(
                    "Failed to decrypt features from GrowthBook API response"
                )
                return None
        elif "features" not in data:
            logger.warning("GrowthBook API response missing features")

        if "encryptedSavedGroups" in data:
            if not decryption_key:
                raise ValueError("Must specify decryption_key")
            try:
                decryptedFeatures = decrypt(
                    data["encryptedSavedGroups"], decryption_key
                )
                data["savedGroups"] = json.loads(decryptedFeatures)
                del data["encryptedSavedGroups"]
                return data
            except Exception:
                logger.warning(
                    "Failed to decrypt saved groups from GrowthBook API response"
                )

        return data

    async def _fetch_features(
        self, api_host: str, client_key: str, decryption_key: str = ""
    ) -> Optional[Dict]:
        decoded = await self._fetch_and_decode(api_host, client_key)
        if not decoded:
            return None

        data = self.decrypt_response(decoded, decryption_key)

        return data

    async def startAutoRefresh(self, api_host, client_key, cb):
        self.sse_client = self.sse_client or SSEClient(
            api_host=api_host, client_key=client_key, on_event=cb
        )
        await self.sse_client.connect_async()

    async def stopAutoRefresh(self):
        await self.sse_client.disconnect_async()

    @staticmethod
    def _get_features_url(api_host: str, client_key: str) -> str:
        api_host = (api_host or "https://cdn.growthbook.io").rstrip("/")
        return api_host + "/api/features/" + client_key


async_feature_repo = AsyncFeatureRepository()


class AsyncGrowthBook(AgnosticGrowthBookBase):
    def __init__(
        self,
        enabled: bool = True,
        attributes: dict = {},
        url: str = "",
        qa_mode: bool = False,
        on_experiment_viewed=None,
        api_host: str = "",
        client_key: str = "",
        decryption_key: str = "",
        cache_ttl: int = 60,
        forced_variations: dict = {},
        sticky_bucket_service: AbstractAsyncStickyBucketService = None,
        sticky_bucket_identifier_attributes: List[str] = None,
        savedGroups: dict = {},
        # Deprecated args
        trackingCallback=None,
        qaMode: bool = False,
        user: dict = {},
        groups: dict = {},
        overrides: dict = {},
        forcedVariations: dict = {},
    ):
        super().__init__(
            enabled=enabled,
            attributes=attributes,
            url=url,
            qa_mode=qa_mode or qaMode,
            on_experiment_viewed=on_experiment_viewed or trackingCallback,
            api_host=api_host,
            client_key=client_key,
            decryption_key=decryption_key,
            cache_ttl=cache_ttl,
            forced_variations=forced_variations or forcedVariations,
            savedGroups=savedGroups,
            sticky_bucket_identifier_attributes=sticky_bucket_identifier_attributes,
            user=user,
            groups=groups,
            overrides=overrides,
        )

        self._global_ctx = AsyncGlobalContext(
            options=self._global_ctx_options,
            sticky_bucket_service=sticky_bucket_service,
            features=self._features,
            saved_groups=self._saved_groups,
        )

        self.sticky_bucket_service = sticky_bucket_service

        self._subscriptions: Set[Any] = set()

    async def load_features(self) -> None:
        if not self._client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        response = await async_feature_repo.load_features(
            self._api_host, self._client_key, self._decryption_key, self._cache_ttl
        )

        if response is not None and "features" in response.keys():
            await self.setFeatures(response["features"])

        if response is not None and "savedGroups" in response:
            self._saved_groups = response["savedGroups"]

    async def _features_event_handler(self, features):
        decoded = json.loads(features)
        if not decoded:
            return None

        data = async_feature_repo.decrypt_response(decoded, self._decryption_key)

        if data is not None:
            if "features" in data:
                await self.set_features(data["features"])
            if "savedGroups" in data:
                self._saved_groups = data["savedGroups"]
            await async_feature_repo.save_in_cache(
                self._client_key, features, self._cache_ttl
            )

    async def _dispatch_sse_event(self, event_data):
        event_type = event_data["type"]
        data = event_data["data"]
        if event_type == "features-updated":
            await self.load_features()
        elif event_type == "features":
            await self._features_event_handler(data)

    async def startAutoRefresh(self):
        if not self._client_key:
            raise ValueError("Must specify `client_key` to start features streaming")

        await async_feature_repo.startAutoRefresh(
            api_host=self._api_host,
            client_key=self._client_key,
            cb=self._dispatch_sse_event,
        )

    async def stopAutoRefresh(self):
        await async_feature_repo.stopAutoRefresh()

    async def set_features(self, features: dict) -> None:
        self._features = {}
        for key, feature in features.items():
            if isinstance(feature, Feature):
                self._features[key] = feature
            else:
                self._features[key] = Feature(
                    rules=feature.get("rules", []),
                    defaultValue=feature.get("defaultValue", None),
                )
        # Update the global context with the new features and saved groups
        self._global_ctx.features = self._features
        self._global_ctx.saved_groups = self._saved_groups
        await self.refresh_sticky_buckets()

    # @deprecated, use set_features
    async def setFeatures(self, features: dict) -> None:
        return await self.set_features(features)

    async def set_attributes(self, attributes: dict) -> None:
        self._attributes = attributes
        await self.refresh_sticky_buckets()

    # @deprecated, use set_attributes
    async def setAttributes(self, attributes: dict) -> None:
        return await self.set_attributes(attributes)

    async def is_on(self, key: str) -> bool:
        return (await self.eval_feature(key)).on

    # @deprecated, use is_on
    async def isOn(self, key: str) -> bool:
        return await self.is_on(key)

    async def is_off(self, key: str) -> bool:
        return (await self.eval_feature(key)).off

    # @deprecated, use is_off
    async def isOff(self, key: str) -> bool:
        return await self.is_off(key)

    async def get_feature_value(self, key: str, fallback):
        res = await self.eval_feature(key)
        return res.value if res.value is not None else fallback

    # @deprecated, use get_feature_value
    async def getFeatureValue(self, key: str, fallback):
        return await self.get_feature_value(key, fallback)

    def _get_eval_context(self) -> AsyncEvaluationContext:
        # use the latest attributes for every evaluation.
        self._user_ctx.attributes = self._attributes
        self._user_ctx.url = self._url
        self._user_ctx.overrides = self._overrides
        # set the url for every evaluation. (unlikely to change)
        self._global_ctx.options.url = self._url
        return AsyncEvaluationContext(
            global_ctx=self._global_ctx,
            user=self._user_ctx,
            stack=StackContext(evaluted_features=set()),
        )

    async def eval_feature(self, key: str) -> FeatureResult:
        return await eval_feature_async(
            key=key,
            evalContext=self._get_eval_context(),
            callback_subscription=self._fireSubscriptions,
        )

    # @deprecated, use eval_feature
    async def evalFeature(self, key: str) -> FeatureResult:
        return await self.eval_feature(key)

    async def _fireSubscriptions(self, experiment: Experiment, result: Result):
        prev = self._assigned.get(experiment.key, None)
        if (
            not prev
            or prev["result"].inExperiment != result.inExperiment
            or prev["result"].variationId != result.variationId
        ):
            self._assigned[experiment.key] = {
                "experiment": experiment,
                "result": result,
            }
            for cb in self._subscriptions:
                try:
                    res = cb(experiment, result)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    pass

    async def run(self, experiment: Experiment) -> Result:
        result = await run_experiment_async(
            experiment=experiment,
            evalContext=self._get_eval_context(),
            tracking_cb=self._track,
        )
        await self._fireSubscriptions(experiment, result)
        return result

    def subscribe(self, callback):
        self._subscriptions.add(callback)
        return lambda: self._subscriptions.remove(callback)

    async def _track(self, experiment: Experiment, result: Result) -> None:
        if not self._trackingCallback:
            return None
        key = (
            result.hashAttribute
            + str(result.hashValue)
            + experiment.key
            + str(result.variationId)
        )
        if not self._tracked.get(key):
            try:
                res = self._trackingCallback(experiment=experiment, result=result)
                if inspect.isawaitable(res):
                    await res
                self._tracked[key] = True
            except Exception:
                pass

    async def refresh_sticky_buckets(self, force: bool = False) -> None:
        if not self.sticky_bucket_service:
            return

        attributes = self._get_sticky_bucket_attributes()
        if not force and attributes == self._sticky_bucket_attributes:
            logger.debug("Skipping refresh of sticky bucket assignments, no changes")
            return

        self._sticky_bucket_attributes = attributes
        self._sticky_bucket_assignment_docs = (
            await self.sticky_bucket_service.get_all_assignments(attributes)
        )
        # Update the user context with the new sticky bucket assignment docs
        self._user_ctx.sticky_bucket_assignment_docs = (
            self._sticky_bucket_assignment_docs
        )

    def _get_sticky_bucket_attributes(self) -> dict:
        attributes: Dict[str, str] = {}
        if self._using_derived_sticky_bucket_attributes:
            self.sticky_bucket_identifier_attributes = (
                self._derive_sticky_bucket_identifier_attributes()
            )

        if not self.sticky_bucket_identifier_attributes:
            return attributes

        for attr in self.sticky_bucket_identifier_attributes:
            _, hash_value = _getHashValue(
                attr=attr, eval_context=self._get_eval_context()
            )
            if hash_value:
                attributes[attr] = hash_value
        return attributes
