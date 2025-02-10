import inspect
import json
import re
from abc import ABC, abstractmethod
from time import time
from typing import Awaitable, Optional, Dict, List, Any, Set, Tuple

import aiohttp
from urllib3 import PoolManager

from . import AgnosticGrowthBookBase
from .growthbook import (
    CacheEntry,
    SSEClient,
    logger,
    decrypt,
    Feature,
    evalCondition,
    FeatureResult,
    inRange,
    gbhash,
    Experiment,
    Result,
    Filter, getQueryStringOverride, inNamespace, getBucketRanges, chooseVariation, VariationMeta
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


class AbstractAsyncStickyBucketService(ABC):
    @abstractmethod
    def get_assignments(self, attributeName: str, attributeValue: str) -> Awaitable[Optional[Dict]]:
        pass

    @abstractmethod
    def save_assignments(self, doc: Dict) -> Awaitable[None]:
        pass

    def get_key(self, attributeName: str, attributeValue: str) -> str:
        return f"{attributeName}||{attributeValue}"

    # By default, just loop through all attributes and call get_assignments
    # Override this method in subclasses to perform a multi-query instead
    async def get_all_assignments(self, attributes: Dict[str, str]) -> Dict[str, Dict]:
        docs = {}
        for attributeName, attributeValue in attributes.items():
            doc = await self.get_assignments(attributeName, attributeValue)
            if doc:
                docs[self.get_key(attributeName, attributeValue)] = doc
        return docs


class AsyncInMemoryStickyBucketService(AbstractAsyncStickyBucketService):
    def __init__(self) -> None:
        self.docs: Dict[str, Dict] = {}

    async def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict]:
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
                        logger.warning("Failed to fetch features, received status code %d", response.status)
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
                data['features'] = json.loads(decryptedFeatures)
                del data['encryptedFeatures']
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
                decryptedFeatures = decrypt(data["encryptedSavedGroups"], decryption_key)
                data['savedGroups'] = json.loads(decryptedFeatures)
                del data['encryptedSavedGroups']
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
        self.sse_client = self.sse_client or SSEClient(api_host=api_host, client_key=client_key, on_event=cb)
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
        sticky_bucket_service: Optional[AbstractAsyncStickyBucketService] = None,
        sticky_bucket_identifier_attributes: Optional[List[str]] = None,
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
            attributes=attributes,
            url=url,
            using_sticky_buckets=sticky_bucket_service is not None,
            sticky_bucket_identifier_attributes=sticky_bucket_identifier_attributes,
            user=user,
        )

        self._enabled = enabled
        self._features: Dict[str, Feature] = {}
        self._saved_groups = savedGroups
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl
        self.sticky_bucket_service = sticky_bucket_service
        self._sticky_bucket_assignment_docs: dict = {}
        self._using_derived_sticky_bucket_attributes = not sticky_bucket_identifier_attributes
        self._sticky_bucket_attributes: Optional[dict] = None

        self._qaMode = qa_mode or qaMode
        self._trackingCallback = on_experiment_viewed or trackingCallback

        # Deprecated args
        self._groups = groups
        self._overrides = overrides
        self._forcedVariations = forced_variations or forcedVariations

        self._tracked: Dict[str, Any] = {}
        self._assigned: Dict[str, Any] = {}
        self._subscriptions: Set[Any] = set()

    async def load_features(self) -> None:
        if not self._client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        features = await async_feature_repo.load_features(
            self._api_host, self._client_key, self._decryption_key, self._cache_ttl
        )

        if features is not None:
            if "features" in features:
                await self.set_features(features["features"])
            if "savedGroups" in features:
                self._saved_groups = features["savedGroups"]
            await async_feature_repo.save_in_cache(self._client_key, features, self._cache_ttl)

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
            await async_feature_repo.save_in_cache(self._client_key, features, self._cache_ttl)

    async def _dispatch_sse_event(self, event_data):
        event_type = event_data['type']
        data = event_data['data']
        if event_type == 'features-updated':
            await self.load_features()
        elif event_type == 'features':
            await self._features_event_handler(data)

    async def startAutoRefresh(self):
        if not self._client_key:
            raise ValueError("Must specify `client_key` to start features streaming")

        await async_feature_repo.startAutoRefresh(
            api_host=self._api_host,
            client_key=self._client_key,
            cb=self._dispatch_sse_event
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
        await self.refresh_sticky_buckets()

    # @deprecated, use set_features
    async def setFeatures(self, features: dict) -> None:
        return await self.set_features(features)

    def get_features(self) -> Dict[str, Feature]:
        return self._features

    async def set_attributes(self, attributes: dict) -> None:
        self._attributes = attributes
        await self.refresh_sticky_buckets()

    # @deprecated, use set_attributes
    async def setAttributes(self, attributes: dict) -> None:
        return await self.set_attributes(attributes)

    def get_attributes(self) -> dict:
        return self._attributes

    def destroy(self) -> None:
        self._subscriptions.clear()
        self._tracked.clear()
        self._assigned.clear()
        self._trackingCallback = None
        self._forcedVariations.clear()
        self._overrides.clear()
        self._groups.clear()
        self._attributes.clear()
        self._features.clear()

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

    async def eval_prereqs(self, parentConditions: List[dict], stack: Set[str]) -> str:
        for parentCondition in parentConditions:
            parentRes = await self._eval_feature(parentCondition.get("id", None), stack)

            if parentRes.source == "cyclicPrerequisite":
                return "cyclic"

            if not evalCondition({'value': parentRes.value}, parentCondition.get("condition", None), self._saved_groups):
                if parentCondition.get("gate", False):
                    return "gate"
                return "fail"
        return "pass"

    async def eval_feature(self, key: str) -> FeatureResult:
        return await self._eval_feature(key, set())

    # @deprecated, use eval_feature
    async def evalFeature(self, key: str) -> FeatureResult:
        return await self.eval_feature(key)

    async def _eval_feature(self, key: str, stack: Set[str]) -> FeatureResult:
        logger.debug("Evaluating feature %s", key)
        if key not in self._features:
            logger.warning("Unknown feature %s", key)
            return FeatureResult(None, "unknownFeature")

        if key in stack:
            logger.warning("Cyclic prerequisite detected, stack: %s", stack)
            return FeatureResult(None, "cyclicPrerequisite")
        stack.add(key)

        feature = self._features[key]
        for rule in feature.rules:
            logger.debug("Evaluating feature %s, rule %s", key, rule.to_dict())
            if (rule.parentConditions):
                prereq_res = await self.eval_prereqs(rule.parentConditions, stack)
                if prereq_res == "gate":
                    logger.debug("Top-level prerequisite failed, return None, feature %s", key)
                    return FeatureResult(None, "prerequisite")
                if prereq_res == "cyclic":
                    # Warning already logged in this case
                    return FeatureResult(None, "cyclicPrerequisite")
                if prereq_res == "fail":
                    logger.debug("Skip rule because of failing prerequisite, feature %s", key)
                    continue

            if rule.condition:
                if not evalCondition(self._attributes, rule.condition, self._saved_groups):
                    logger.debug(
                        "Skip rule because of failed condition, feature %s", key
                    )
                    continue
            if rule.filters:
                if self._isFilteredOut(rule.filters):
                    logger.debug(
                        "Skip rule because of filters/namespaces, feature %s", key
                    )
                    continue
            if rule.force is not None:
                if not self._isIncludedInRollout(
                    rule.seed or key,
                    rule.hashAttribute,
                    rule.fallbackAttribute,
                    rule.range,
                    rule.coverage,
                    rule.hashVersion,
                ):
                    logger.debug(
                        "Skip rule because user not included in percentage rollout, feature %s",
                        key,
                    )
                    continue

                logger.debug("Force value from rule, feature %s", key)
                return FeatureResult(rule.force, "force", ruleId=rule.id)

            if rule.variations is None:
                logger.warning("Skip invalid rule, feature %s", key)
                continue

            exp = Experiment(
                key=rule.key or key,
                variations=rule.variations,
                coverage=rule.coverage,
                weights=rule.weights,
                hashAttribute=rule.hashAttribute,
                fallbackAttribute=rule.fallbackAttribute,
                namespace=rule.namespace,
                hashVersion=rule.hashVersion,
                meta=rule.meta,
                ranges=rule.ranges,
                name=rule.name,
                phase=rule.phase,
                seed=rule.seed,
                filters=rule.filters,
                condition=rule.condition,
                disableStickyBucketing=rule.disableStickyBucketing,
                bucketVersion=rule.bucketVersion,
                minBucketVersion=rule.minBucketVersion,
            )

            result = await self._run(exp, key)
            await self._fireSubscriptions(exp, result)

            if not result.inExperiment:
                logger.debug(
                    "Skip rule because user not included in experiment, feature %s", key
                )
                continue

            if result.passthrough:
                logger.debug("Continue to next rule, feature %s", key)
                continue

            logger.debug("Assign value from experiment, feature %s", key)
            return FeatureResult(
                result.value, "experiment", exp, result, ruleId=rule.id
            )

        logger.debug("Use default value for feature %s", key)
        return FeatureResult(feature.defaultValue, "defaultValue")

    def get_all_results(self):
        return self._assigned.copy()

    def _getOrigHashValue(self, attr: Optional[str] = None, fallbackAttr: Optional[str] = None) -> Tuple[str, str]:
        attr = attr or "id"
        val = ""
        if attr in self._attributes:
            val = "" if self._attributes[attr] is None else self._attributes[attr]
        elif attr in self._user:
            val = "" if self._user[attr] is None else self._user[attr]

        # If no match, try fallback
        if (not val or val == "") and fallbackAttr and self.sticky_bucket_service:
            if fallbackAttr in self._attributes:
                val = "" if self._attributes[fallbackAttr] is None else self._attributes[fallbackAttr]
            elif fallbackAttr in self._user:
                val = "" if self._user[fallbackAttr] is None else self._user[fallbackAttr]

            if not val or val != "":
                attr = fallbackAttr

        return (attr, val)

    def _getHashValue(self, attr: Optional[str] = None, fallbackAttr: Optional[str] = None) -> Tuple[str, str]:
        (attr, val) = self._getOrigHashValue(attr, fallbackAttr)
        return (attr, str(val))

    def _isIncludedInRollout(
        self,
        seed: str,
        hashAttribute: Optional[str] = None,
        fallbackAttribute: Optional[str] = None,
        range: Optional[Tuple[float, float]] = None,
        coverage: Optional[float] = None,
        hashVersion: Optional[int] = None,
    ) -> bool:
        if coverage is None and range is None:
            return True

        (_, hash_value) = self._getHashValue(hashAttribute, fallbackAttribute)
        if hash_value == "":
            return False

        n = gbhash(seed, hash_value, hashVersion or 1)
        if n is None:
            return False

        if range:
            return inRange(n, range)
        elif coverage is not None:
            return n <= coverage

        return True

    def _isFilteredOut(self, filters: List[Filter]) -> bool:
        for filter in filters:
            (_, hash_value) = self._getHashValue(filter.get("attribute", "id"))
            if hash_value == "":
                return False

            n = gbhash(filter.get("seed", ""), hash_value, filter.get("hashVersion", 2))
            if n is None:
                return False

            filtered = False
            for range in filter["ranges"]:
                if inRange(n, range):
                    filtered = True
                    break
            if not filtered:
                return True
        return False

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
        result = await self._run(experiment)
        await self._fireSubscriptions(experiment, result)
        return result

    def subscribe(self, callback):
        self._subscriptions.add(callback)
        return lambda: self._subscriptions.remove(callback)

    async def _run(self, experiment: Experiment, featureId: Optional[str] = None) -> Result:
        # 1. If experiment has less than 2 variations, return immediately
        if len(experiment.variations) < 2:
            logger.warning(
                "Experiment %s has less than 2 variations, skip", experiment.key
            )
            return self._getExperimentResult(experiment, featureId=featureId)
        # 2. If growthbook is disabled, return immediately
        if not self._enabled:
            logger.debug(
                "Skip experiment %s because GrowthBook is disabled", experiment.key
            )
            return self._getExperimentResult(experiment, featureId=featureId)
        # 2.5. If the experiment props have been overridden, merge them in
        if self._overrides.get(experiment.key, None):
            experiment.update(self._overrides[experiment.key])
        # 3. If experiment is forced via a querystring in the url
        qs = getQueryStringOverride(
            experiment.key, self._url, len(experiment.variations)
        )
        if qs is not None:
            logger.debug(
                "Force variation %d from URL querystring, experiment %s",
                qs,
                experiment.key,
            )
            return self._getExperimentResult(experiment, qs, featureId=featureId)
        # 4. If variation is forced in the context
        if self._forcedVariations.get(experiment.key, None) is not None:
            logger.debug(
                "Force variation %d from GrowthBook context, experiment %s",
                self._forcedVariations[experiment.key],
                experiment.key,
            )
            return self._getExperimentResult(
                experiment, self._forcedVariations[experiment.key], featureId=featureId
            )
        # 5. If experiment is a draft or not active, return immediately
        if experiment.status == "draft" or not experiment.active:
            logger.debug("Experiment %s is not active, skip", experiment.key)
            return self._getExperimentResult(experiment, featureId=featureId)

        # 6. Get the user hash attribute and value
        (hashAttribute, hashValue) = self._getHashValue(experiment.hashAttribute, experiment.fallbackAttribute)
        if not hashValue:
            logger.debug(
                "Skip experiment %s because user's hashAttribute value is empty",
                experiment.key,
            )
            return self._getExperimentResult(experiment, featureId=featureId)

        assigned = -1

        found_sticky_bucket = False
        sticky_bucket_version_is_blocked = False
        if self.sticky_bucket_service and not experiment.disableStickyBucketing:
            sticky_bucket = self._get_sticky_bucket_variation(
                experiment.key,
                experiment.bucketVersion,
                experiment.minBucketVersion,
                experiment.meta,
                hash_attribute=experiment.hashAttribute,
                fallback_attribute=experiment.fallbackAttribute,
            )
            found_sticky_bucket = sticky_bucket.get('variation', 0) >= 0
            assigned = sticky_bucket.get('variation', 0)
            sticky_bucket_version_is_blocked = sticky_bucket.get('versionIsBlocked', False)

        if found_sticky_bucket:
            logger.debug("Found sticky bucket for experiment %s, assigning sticky variation %s", experiment.key, assigned)

        # Some checks are not needed if we already have a sticky bucket
        if not found_sticky_bucket:
            # 7. Filtered out / not in namespace
            if experiment.filters:
                if self._isFilteredOut(experiment.filters):
                    logger.debug(
                        "Skip experiment %s because of filters/namespaces", experiment.key
                    )
                    return self._getExperimentResult(experiment, featureId=featureId)
            elif experiment.namespace and not inNamespace(hashValue, experiment.namespace):
                logger.debug("Skip experiment %s because of namespace", experiment.key)
                return self._getExperimentResult(experiment, featureId=featureId)

            # 7.5. If experiment has an include property
            if experiment.include:
                try:
                    if not experiment.include():
                        logger.debug(
                            "Skip experiment %s because include() returned false",
                            experiment.key,
                        )
                        return self._getExperimentResult(experiment, featureId=featureId)
                except Exception:
                    logger.warning(
                        "Skip experiment %s because include() raised an Exception",
                        experiment.key,
                    )
                    return self._getExperimentResult(experiment, featureId=featureId)

            # 8. Exclude if condition is false
            if experiment.condition and not evalCondition(
                self._attributes, experiment.condition, self._saved_groups
            ):
                logger.debug(
                    "Skip experiment %s because user failed the condition", experiment.key
                )
                return self._getExperimentResult(experiment, featureId=featureId)

            # 8.05 Exclude if parent conditions are not met
            if (experiment.parentConditions):
                prereq_res = await self.eval_prereqs(experiment.parentConditions, set())
                if prereq_res == "gate" or prereq_res == "fail":
                    logger.debug("Skip experiment %s because of failing prerequisite", experiment.key)
                    return self._getExperimentResult(experiment, featureId=featureId)
                if prereq_res == "cyclic":
                    logger.debug("Skip experiment %s because of cyclic prerequisite", experiment.key)
                    return self._getExperimentResult(experiment, featureId=featureId)

            # 8.1. Make sure user is in a matching group
            if experiment.groups and len(experiment.groups):
                expGroups = self._groups or {}
                matched = False
                for group in experiment.groups:
                    if expGroups[group]:
                        matched = True
                if not matched:
                    logger.debug(
                        "Skip experiment %s because user not in required group",
                        experiment.key,
                    )
                    return self._getExperimentResult(experiment, featureId=featureId)

        # The following apply even when in a sticky bucket

        # 8.2. If experiment.url is set, see if it's valid
        if experiment.url:
            if not self._urlIsValid(experiment.url):
                logger.debug(
                    "Skip experiment %s because current URL is not targeted",
                    experiment.key,
                )
                return self._getExperimentResult(experiment, featureId=featureId)

        # 9. Get bucket ranges and choose variation
        n = gbhash(
            experiment.seed or experiment.key, hashValue, experiment.hashVersion or 1
        )
        if n is None:
            logger.warning(
                "Skip experiment %s because of invalid hashVersion", experiment.key
            )
            return self._getExperimentResult(experiment, featureId=featureId)

        if not found_sticky_bucket:
            c = experiment.coverage
            ranges = experiment.ranges or getBucketRanges(
                len(experiment.variations), c if c is not None else 1, experiment.weights
            )
            assigned = chooseVariation(n, ranges)

        # Unenroll if any prior sticky buckets are blocked by version
        if sticky_bucket_version_is_blocked:
            logger.debug("Skip experiment %s because sticky bucket version is blocked", experiment.key)
            return self._getExperimentResult(experiment, featureId=featureId, stickyBucketUsed=True)

        # 10. Return if not in experiment
        if assigned < 0:
            logger.debug(
                "Skip experiment %s because user is not included in the rollout",
                experiment.key,
            )
            return self._getExperimentResult(experiment, featureId=featureId)

        # 11. If experiment is forced, return immediately
        if experiment.force is not None:
            logger.debug(
                "Force variation %d in experiment %s", experiment.force, experiment.key
            )
            return self._getExperimentResult(
                experiment, experiment.force, featureId=featureId
            )

        # 12. Exclude if in QA mode
        if self._qaMode:
            logger.debug("Skip experiment %s because of QA Mode", experiment.key)
            return self._getExperimentResult(experiment, featureId=featureId)

        # 12.5. If experiment is stopped, return immediately
        if experiment.status == "stopped":
            logger.debug("Skip experiment %s because it is stopped", experiment.key)
            return self._getExperimentResult(experiment, featureId=featureId)

        # 13. Build the result object
        result = self._getExperimentResult(
            experiment, assigned, True, featureId=featureId, bucket=n, stickyBucketUsed=found_sticky_bucket
        )

        # 13.5 Persist sticky bucket
        if self.sticky_bucket_service and not experiment.disableStickyBucketing:
            assignment = {}
            assignment[self._get_sticky_bucket_experiment_key(
                experiment.key,
                experiment.bucketVersion
            )] = result.key

            data = self._generate_sticky_bucket_assignment_doc(
                hashAttribute,
                hashValue,
                assignment
            )
            doc = data.get("doc", None)
            if doc and data.get('changed', False):
                if not self._sticky_bucket_assignment_docs:
                    self._sticky_bucket_assignment_docs = {}
                self._sticky_bucket_assignment_docs[data.get('key')] = doc
                await self.sticky_bucket_service.save_assignments(doc)

        # 14. Fire the tracking callback if set
        await self._track(experiment, result)

        # 15. Return the result
        logger.debug("Assigned variation %d in experiment %s", assigned, experiment.key)
        return result

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

    def _urlIsValid(self, pattern) -> bool:
        if not self._url:
            return False

        try:
            r = re.compile(pattern)
            if r.search(self._url):
                return True

            pathOnly = re.sub(r"^[^/]*/", "/", re.sub(r"^https?:\/\/", "", self._url))
            if r.search(pathOnly):
                return True
            return False
        except Exception:
            return True

    def _getExperimentResult(
        self,
        experiment: Experiment,
        variationId: int = -1,
        hashUsed: bool = False,
        featureId: Optional[str] = None,
        bucket: Optional[float] = None,
        stickyBucketUsed: bool = False
    ) -> Result:
        inExperiment = True
        if variationId < 0 or variationId > len(experiment.variations) - 1:
            variationId = 0
            inExperiment = False

        meta = None
        if experiment.meta:
            meta = experiment.meta[variationId]

        (hashAttribute, hashValue) = self._getOrigHashValue(experiment.hashAttribute, experiment.fallbackAttribute)

        return Result(
            featureId=featureId,
            inExperiment=inExperiment,
            variationId=variationId,
            value=experiment.variations[variationId],
            hashUsed=hashUsed,
            hashAttribute=hashAttribute,
            hashValue=hashValue,
            meta=meta,
            bucket=bucket,
            stickyBucketUsed=stickyBucketUsed
        )

    def _derive_sticky_bucket_identifier_attributes(self) -> List[str]:
        attributes = set()
        for key, feature in self._features.items():
            for rule in feature.rules:
                if rule.variations:
                    attributes.add(rule.hashAttribute or "id")
                    if rule.fallbackAttribute:
                        attributes.add(rule.fallbackAttribute)
        return list(attributes)

    def _get_sticky_bucket_attributes(self) -> dict:
        attributes: Dict[str, str] = {}
        if self._using_derived_sticky_bucket_attributes:
            self.sticky_bucket_identifier_attributes = self._derive_sticky_bucket_identifier_attributes()

        if not self.sticky_bucket_identifier_attributes:
            return attributes

        for attr in self.sticky_bucket_identifier_attributes:
            _, hash_value = self._getHashValue(attr)
            if hash_value:
                attributes[attr] = hash_value
        return attributes

    def _get_sticky_bucket_assignments(self, attr: Optional[str] = None, fallback: Optional[str] = None) -> Dict[str, str]:
        merged: Dict[str, str] = {}

        _, hashValue = self._getHashValue(attr)
        key = f"{attr}||{hashValue}"
        if key in self._sticky_bucket_assignment_docs:
            merged = self._sticky_bucket_assignment_docs[key].get("assignments", {})

        if fallback:
            _, hashValue = self._getHashValue(fallback)
            key = f"{fallback}||{hashValue}"
            if key in self._sticky_bucket_assignment_docs:
                # Merge the fallback assignments, but don't overwrite existing ones
                for k, v in self._sticky_bucket_assignment_docs[key].get("assignments", {}).items():
                    if k not in merged:
                        merged[k] = v

        return merged

    def _is_blocked(
        self,
        assignments: Dict[str, str],
        experiment_key: str,
        min_bucket_version: int
    ) -> bool:
        if min_bucket_version > 0:
            for i in range(min_bucket_version):
                blocked_key = self._get_sticky_bucket_experiment_key(experiment_key, i)
                if blocked_key in assignments:
                    return True
        return False

    def _get_sticky_bucket_variation(
        self,
        experiment_key: str,
        bucket_version: Optional[int] = None,
        min_bucket_version: Optional[int] = None,
        meta: Optional[List[VariationMeta]] = None,
        hash_attribute: Optional[str] = None,
        fallback_attribute: Optional[str] = None
    ) -> dict:
        bucket_version = bucket_version or 0
        min_bucket_version = min_bucket_version or 0
        meta = meta or []

        id = self._get_sticky_bucket_experiment_key(experiment_key, bucket_version)

        assignments = self._get_sticky_bucket_assignments(hash_attribute, fallback_attribute)
        if self._is_blocked(assignments, experiment_key, min_bucket_version):
            return {
                'variation': -1,
                'versionIsBlocked': True
            }

        variation_key = assignments.get(id, None)
        if not variation_key:
            return {
                'variation': -1
            }

        # Find the key in meta
        variation = next((i for i, v in enumerate(meta) if v.get("key") == variation_key), -1)
        if variation < 0:
            return {
                'variation': -1
            }

        return {'variation': variation}

    def _get_sticky_bucket_experiment_key(self, experiment_key: str, bucket_version: int = 0) -> str:
        return experiment_key + "__" + str(bucket_version)

    async def refresh_sticky_buckets(self, force: bool = False) -> None:
        if not self.sticky_bucket_service:
            return

        attributes = self._get_sticky_bucket_attributes()
        if not force and attributes == self._sticky_bucket_attributes:
            logger.debug("Skipping refresh of sticky bucket assignments, no changes")
            return

        self._sticky_bucket_attributes = attributes
        self._sticky_bucket_assignment_docs = await self.sticky_bucket_service.get_all_assignments(attributes)

    def _generate_sticky_bucket_assignment_doc(self, attribute_name: str, attribute_value: str, assignments: dict):
        key = attribute_name + "||" + attribute_value
        existing_assignments = self._sticky_bucket_assignment_docs.get(key, {}).get("assignments", {})

        new_assignments = {**existing_assignments, **assignments}

        # Compare JSON strings to see if they have changed
        existing_json = json.dumps(existing_assignments, sort_keys=True)
        new_json = json.dumps(new_assignments, sort_keys=True)
        changed = existing_json != new_json

        return {
            'key': key,
            'doc': {
                'attributeName': attribute_name,
                'attributeValue': attribute_value,
                'assignments': new_assignments
            },
            'changed': changed
        }
