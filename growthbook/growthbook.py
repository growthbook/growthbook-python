#!/usr/bin/env python
"""
This is the Python client library for GrowthBook, the open-source
feature flagging and A/B testing platform.
More info at https://www.growthbook.io
"""

import sys
import json
import threading
import logging

from abc import ABC, abstractmethod
from typing import Optional, Any, Set, Tuple, List, Dict, Callable

from .common_types import ( EvaluationContext, 
    Experiment, 
    FeatureResult, 
    Feature,
    GlobalContext, 
    Options, 
    Result, StackContext, 
    UserContext, 
    AbstractStickyBucketService,
    FeatureRule
)

# Only require typing_extensions if using Python 3.7 or earlier
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from base64 import b64decode
from time import time
import aiohttp
import asyncio

from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError, ClientPayloadError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from urllib3 import PoolManager

from .core import _getHashValue, eval_feature as core_eval_feature, run_experiment

logger = logging.getLogger("growthbook")

def decrypt(encrypted_str: str, key_str: str) -> str:
    iv_str, ct_str = encrypted_str.split(".", 2)

    key = b64decode(key_str)
    iv = b64decode(iv_str)
    ct = b64decode(ct_str)

    cipher = Cipher(algorithms.AES128(key), modes.CBC(iv))
    decryptor = cipher.decryptor()

    decrypted = decryptor.update(ct) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    bytestring = unpadder.update(decrypted) + unpadder.finalize()

    return bytestring.decode("utf-8")

class AbstractFeatureCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def set(self, key: str, value: Dict, ttl: int) -> None:
        pass

    def clear(self) -> None:
        pass


class CacheEntry(object):
    def __init__(self, value: Dict, ttl: int) -> None:
        self.value = value
        self.ttl = ttl
        self.expires = time() + ttl

    def update(self, value: Dict):
        self.value = value
        self.expires = time() + self.ttl


class InMemoryFeatureCache(AbstractFeatureCache):
    def __init__(self) -> None:
        self.cache: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            entry = self.cache[key]
            if entry.expires >= time():
                return entry.value
        return None

    def set(self, key: str, value: Dict, ttl: int) -> None:
        if key in self.cache:
            self.cache[key].update(value)
        self.cache[key] = CacheEntry(value, ttl)

    def clear(self) -> None:
        self.cache.clear()

class InMemoryStickyBucketService(AbstractStickyBucketService):
    def __init__(self) -> None:
        self.docs: Dict[str, Dict] = {}

    def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict]:
        return self.docs.get(self.get_key(attributeName, attributeValue), None)

    def save_assignments(self, doc: Dict) -> None:
        self.docs[self.get_key(doc["attributeName"], doc["attributeValue"])] = doc

    def destroy(self) -> None:
        self.docs.clear()


class SSEClient:
    def __init__(self, api_host, client_key, on_event, reconnect_delay=5, headers=None):
        self.api_host = api_host
        self.client_key = client_key

        self.on_event = on_event
        self.reconnect_delay = reconnect_delay

        self._sse_session = None
        self._sse_thread = None
        self._loop = None

        self.is_running = False

        self.headers = {
            "Accept": "application/json; q=0.5, text/event-stream",
            "Cache-Control": "no-cache",
        }

        if headers:
            self.headers.update(headers)

    def connect(self):
        if self.is_running:
            logger.debug("Streaming session is already running.")
            return

        self.is_running = True
        self._sse_thread = threading.Thread(target=self._run_sse_channel)
        self._sse_thread.start()

    def disconnect(self):
        self.is_running = False
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._stop_session(), self._loop)
            try:
                future.result()
            except Exception as e:
                logger.error(f"Streaming disconnect error: {e}")

        if self._sse_thread:
            self._sse_thread.join(timeout=5)

        logger.debug("Streaming session disconnected")

    def _get_sse_url(self, api_host: str, client_key: str) -> str:
        api_host = (api_host or "https://cdn.growthbook.io").rstrip("/")
        return f"{api_host}/sub/{client_key}"

    async def _init_session(self):
        url = self._get_sse_url(self.api_host, self.client_key)
        
        while self.is_running:
            try:
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    self._sse_session = session

                    async with session.get(url) as response:
                        response.raise_for_status()
                        await self._process_response(response)
            except ClientResponseError as e:
                logger.error(f"Streaming error, closing connection: {e.status} {e.message}")
                self.is_running = False
                break
            except (ClientConnectorError, ClientPayloadError) as e:
                logger.error(f"Streaming error: {e}")
                if not self.is_running:
                    break
                await self._wait_for_reconnect()
            except TimeoutError:
                logger.warning(f"Streaming connection timed out after {self.timeout} seconds.")
                await self._wait_for_reconnect()
            except asyncio.CancelledError:
                logger.debug("Streaming was cancelled.")
                break
            finally:
                await self._close_session()

    async def _process_response(self, response):
        event_data = {}
        async for line in response.content:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line.startswith("event:"):
                event_data['type'] = decoded_line[len("event:"):].strip()
            elif decoded_line.startswith("data:"):
                event_data['data'] = event_data.get('data', '') + f"\n{decoded_line[len('data:'):].strip()}"
            elif not decoded_line:
                if 'type' in event_data and 'data' in event_data:
                    self.on_event(event_data)
                event_data = {}

        if 'type' in event_data and 'data' in event_data:
            self.on_event(event_data)

    async def _wait_for_reconnect(self):
        logger.debug(f"Attempting to reconnect streaming in {self.reconnect_delay}")
        await asyncio.sleep(self.reconnect_delay)

    async def _close_session(self):
        if self._sse_session:
            await self._sse_session.close()
            logger.debug("Streaming session closed.")

    def _run_sse_channel(self):
        self._loop = asyncio.new_event_loop()
        
        try:
            self._loop.run_until_complete(self._init_session())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _stop_session(self):
        if self._sse_session:
            await self._sse_session.close()

        if self._loop and self._loop.is_running():
            tasks = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

class FeatureRepository(object):
    def __init__(self) -> None:
        self.cache: AbstractFeatureCache = InMemoryFeatureCache()
        self.http: Optional[PoolManager] = None
        self.sse_client: Optional[SSEClient] = None
        self._feature_update_callbacks: List[Callable[[Dict], None]] = []

    def set_cache(self, cache: AbstractFeatureCache) -> None:
        self.cache = cache

    def clear_cache(self):
        self.cache.clear()

    def save_in_cache(self, key: str, res, ttl: int = 600):
        self.cache.set(key, res, ttl)

    def add_feature_update_callback(self, callback: Callable[[Dict], None]) -> None:
        """Add a callback to be notified when features are updated due to cache expiry"""
        if callback not in self._feature_update_callbacks:
            self._feature_update_callbacks.append(callback)

    def remove_feature_update_callback(self, callback: Callable[[Dict], None]) -> None:
        """Remove a feature update callback"""
        if callback in self._feature_update_callbacks:
            self._feature_update_callbacks.remove(callback)

    def _notify_feature_update_callbacks(self, features_data: Dict) -> None:
        """Notify all registered callbacks about feature updates"""
        for callback in self._feature_update_callbacks:
            try:
                callback(features_data)
            except Exception as e:
                logger.warning(f"Error in feature update callback: {e}")

    # Loads features with an in-memory cache in front using stale-while-revalidate approach
    def load_features(
        self, api_host: str, client_key: str, decryption_key: str = "", ttl: int = 600
    ) -> Optional[Dict]:
        if not client_key:
            raise ValueError("Must specify `client_key` to refresh features")
        
        key = api_host + "::" + client_key

        cached = self.cache.get(key)
        if not cached:
            res = self._fetch_features(api_host, client_key, decryption_key)
            if res is not None:
                self.cache.set(key, res, ttl)
                logger.debug("Fetched features from API, stored in cache")
                # Notify callbacks about fresh features
                self._notify_feature_update_callbacks(res)
                return res
        return cached
    
    async def load_features_async(
        self, api_host: str, client_key: str, decryption_key: str = "", ttl: int = 600
    ) -> Optional[Dict]:
        key = api_host + "::" + client_key

        cached = self.cache.get(key)
        if not cached:
            res = await self._fetch_features_async(api_host, client_key, decryption_key)
            if res is not None:
                self.cache.set(key, res, ttl)
                logger.debug("Fetched features from API, stored in cache")
                # Notify callbacks about fresh features
                self._notify_feature_update_callbacks(res)
                return res
        return cached

    # Perform the GET request (separate method for easy mocking)
    def _get(self, url: str):
        self.http = self.http or PoolManager()
        return self.http.request("GET", url)
    
    def _fetch_and_decode(self, api_host: str, client_key: str) -> Optional[Dict]:
        try:
            r = self._get(self._get_features_url(api_host, client_key))
            if r.status >= 400:
                logger.warning(
                    "Failed to fetch features, received status code %d", r.status
                )
                return None
            decoded = json.loads(r.data.decode("utf-8"))
            return decoded
        except Exception:
            logger.warning("Failed to decode feature JSON from GrowthBook API")
            return None
        
    async def _fetch_and_decode_async(self, api_host: str, client_key: str) -> Optional[Dict]:
        try:
            url = self._get_features_url(api_host, client_key)
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

    # Fetch features from the GrowthBook API
    def _fetch_features(
        self, api_host: str, client_key: str, decryption_key: str = ""
    ) -> Optional[Dict]:
        decoded = self._fetch_and_decode(api_host, client_key)
        if not decoded:
            return None

        data = self.decrypt_response(decoded, decryption_key)

        return data
        
    async def _fetch_features_async(
        self, api_host: str, client_key: str, decryption_key: str = ""
    ) -> Optional[Dict]:
        decoded = await self._fetch_and_decode_async(api_host, client_key)
        if not decoded:
            return None

        data = self.decrypt_response(decoded, decryption_key)

        return data


    def startAutoRefresh(self, api_host, client_key, cb):
        if not client_key:
            raise ValueError("Must specify `client_key` to start features streaming")
        self.sse_client = self.sse_client or SSEClient(api_host=api_host, client_key=client_key, on_event=cb)
        self.sse_client.connect()

    def stopAutoRefresh(self):
        self.sse_client.disconnect()

    @staticmethod
    def _get_features_url(api_host: str, client_key: str) -> str:
        api_host = (api_host or "https://cdn.growthbook.io").rstrip("/")
        return api_host + "/api/features/" + client_key


# Singleton instance
feature_repo = FeatureRepository()

class GrowthBook(object):
    def __init__(
        self,
        enabled: bool = True,
        attributes: dict = {},
        url: str = "",
        features: dict = {},
        qa_mode: bool = False,
        on_experiment_viewed=None,
        api_host: str = "",
        client_key: str = "",
        decryption_key: str = "",
        cache_ttl: int = 600,
        forced_variations: dict = {},
        sticky_bucket_service: AbstractStickyBucketService = None,
        sticky_bucket_identifier_attributes: List[str] = None,
        savedGroups: dict = {},
        streaming: bool = False,
        # Deprecated args
        trackingCallback=None,
        qaMode: bool = False,
        user: dict = {},
        groups: dict = {},
        overrides: dict = {},
        forcedVariations: dict = {},
    ):
        self._enabled = enabled
        self._attributes = attributes
        self._url = url
        self._features: Dict[str, Feature] = {}
        self._saved_groups = savedGroups
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl
        self.sticky_bucket_identifier_attributes = sticky_bucket_identifier_attributes
        self.sticky_bucket_service = sticky_bucket_service
        self._sticky_bucket_assignment_docs: dict = {}
        self._using_derived_sticky_bucket_attributes = not sticky_bucket_identifier_attributes
        self._sticky_bucket_attributes: Optional[dict] = None

        self._qaMode = qa_mode or qaMode
        self._trackingCallback = on_experiment_viewed or trackingCallback

        self._streaming = streaming

        # Deprecated args
        self._user = user
        self._groups = groups
        self._overrides = overrides
        self._forcedVariations = forced_variations or forcedVariations

        self._tracked: Dict[str, Any] = {}
        self._assigned: Dict[str, Any] = {}
        self._subscriptions: Set[Any] = set()

        self._global_ctx = GlobalContext(
            options=Options(
                url=self._url,
                api_host=self._api_host,
                client_key=self._client_key,
                decryption_key=self._decryption_key,
                cache_ttl=self._cache_ttl,
                sticky_bucket_service=self.sticky_bucket_service,
                sticky_bucket_identifier_attributes=self.sticky_bucket_identifier_attributes,
                enabled=self._enabled,
                qa_mode=self._qaMode
            ),
            features={},
            saved_groups=self._saved_groups
        )       
        # Create a user context for the current user
        self._user_ctx: UserContext = UserContext(
            url=self._url,
            attributes=self._attributes,
            groups=self._groups,
            forced_variations=self._forcedVariations,
            overrides=self._overrides,
            sticky_bucket_assignment_docs=self._sticky_bucket_assignment_docs
        )

        if features:
            self.setFeatures(features)

        # Register for automatic feature updates when cache expires
        if self._client_key:
            feature_repo.add_feature_update_callback(self._on_feature_update)

        if self._streaming:
            self.load_features()
            self.startAutoRefresh()

    def _on_feature_update(self, features_data: Dict) -> None:
        """Callback to handle automatic feature updates from FeatureRepository"""
        if features_data and "features" in features_data:
            self.set_features(features_data["features"])
        if features_data and "savedGroups" in features_data:
            self._saved_groups = features_data["savedGroups"]

    def load_features(self) -> None:

        response = feature_repo.load_features(
            self._api_host, self._client_key, self._decryption_key, self._cache_ttl
        )
        if response is not None and "features" in response.keys():
            self.setFeatures(response["features"])

        if response is not None and "savedGroups" in response:
            self._saved_groups = response["savedGroups"]

    async def load_features_async(self) -> None:
        if not self._client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        features = await feature_repo.load_features_async(
            self._api_host, self._client_key, self._decryption_key, self._cache_ttl
        )

        if features is not None:
            if "features" in features:
                self.setFeatures(features["features"])
            if "savedGroups" in features:
                self._saved_groups = features["savedGroups"]
            feature_repo.save_in_cache(self._client_key, features, self._cache_ttl)

    def _features_event_handler(self, features):
        decoded = json.loads(features)
        if not decoded:
            return None
        
        data = feature_repo.decrypt_response(decoded, self._decryption_key)

        if data is not None:
            if "features" in data:
                self.setFeatures(data["features"])
            if "savedGroups" in data:
                self._saved_groups = data["savedGroups"]
            feature_repo.save_in_cache(self._client_key, features, self._cache_ttl)

    def _dispatch_sse_event(self, event_data):
        event_type = event_data['type']
        data = event_data['data']
        if event_type == 'features-updated':
            self.load_features()
        elif event_type == 'features':
            self._features_event_handler(data)


    def startAutoRefresh(self):
        if not self._client_key:
            raise ValueError("Must specify `client_key` to start features streaming")
       
        feature_repo.startAutoRefresh(
            api_host=self._api_host, 
            client_key=self._client_key,
            cb=self._dispatch_sse_event
        )

    def stopAutoRefresh(self):
        feature_repo.stopAutoRefresh()

    # @deprecated, use set_features
    def setFeatures(self, features: dict) -> None:
        return self.set_features(features)

    def set_features(self, features: dict) -> None:
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
        self.refresh_sticky_buckets()

    # @deprecated, use get_features
    def getFeatures(self) -> Dict[str, Feature]:
        return self.get_features()

    def get_features(self) -> Dict[str, Feature]:
        return self._features

    # @deprecated, use set_attributes
    def setAttributes(self, attributes: dict) -> None:
        return self.set_attributes(attributes)

    def set_attributes(self, attributes: dict) -> None:
        self._attributes = attributes
        self.refresh_sticky_buckets()

    # @deprecated, use get_attributes
    def getAttributes(self) -> dict:
        return self.get_attributes()

    def get_attributes(self) -> dict:
        return self._attributes

    def destroy(self) -> None:
        # Clean up feature update callback
        if self._client_key:
            feature_repo.remove_feature_update_callback(self._on_feature_update)
            
        self._subscriptions.clear()
        self._tracked.clear()
        self._assigned.clear()
        self._trackingCallback = None
        self._forcedVariations.clear()
        self._overrides.clear()
        self._groups.clear()
        self._attributes.clear()
        self._features.clear()

    # @deprecated, use is_on
    def isOn(self, key: str) -> bool:
        return self.is_on(key)

    def is_on(self, key: str) -> bool:
        return self.evalFeature(key).on

    # @deprecated, use is_off
    def isOff(self, key: str) -> bool:
        return self.is_off(key)

    def is_off(self, key: str) -> bool:
        return self.evalFeature(key).off

    # @deprecated, use get_feature_value
    def getFeatureValue(self, key: str, fallback):
        return self.get_feature_value(key, fallback)

    def get_feature_value(self, key: str, fallback):
        res = self.evalFeature(key)
        return res.value if res.value is not None else fallback

    # @deprecated, use eval_feature
    def evalFeature(self, key: str) -> FeatureResult:
        return self.eval_feature(key)
    
    def _ensure_fresh_features(self) -> None:
        """Lazy refresh: Check cache expiry and refresh if needed, but only if client_key is provided"""
        
        if self._streaming or not self._client_key:
            return  # Skip cache checks - SSE handles freshness for streaming users

        try:
            self.load_features()
        except Exception as e:
            logger.warning(f"Failed to refresh features: {e}")

    def _get_eval_context(self) -> EvaluationContext:
        # Lazy refresh: ensure features are fresh before evaluation
        self._ensure_fresh_features()
        
        # use the latest attributes for every evaluation.
        self._user_ctx.attributes = self._attributes
        self._user_ctx.url = self._url
        self._user_ctx.overrides = self._overrides
        # set the url for every evaluation. (unlikely to change)
        self._global_ctx.options.url = self._url
        return EvaluationContext(
            global_ctx = self._global_ctx,
            user = self._user_ctx,
            stack = StackContext(evaluated_features=set())
        )

    def eval_feature(self, key: str) -> FeatureResult:
        return core_eval_feature(key=key, 
                                 evalContext=self._get_eval_context(), 
                                 callback_subscription=self._fireSubscriptions
                                 )

    # @deprecated, use get_all_results
    def getAllResults(self):
        return self.get_all_results()

    def get_all_results(self):
        return self._assigned.copy()

    def _fireSubscriptions(self, experiment: Experiment, result: Result):
        if experiment is None:
            return
        
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
                    cb(experiment, result)
                except Exception:
                    pass

    def run(self, experiment: Experiment) -> Result:
        # result = self._run(experiment)
        result = run_experiment(experiment=experiment, 
                                evalContext=self._get_eval_context(),
                                tracking_cb=self._track
                                )

        self._fireSubscriptions(experiment, result)
        return result

    def subscribe(self, callback):
        self._subscriptions.add(callback)
        return lambda: self._subscriptions.remove(callback)

    def _track(self, experiment: Experiment, result: Result) -> None:
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
                self._trackingCallback(experiment=experiment, result=result)
                self._tracked[key] = True
            except Exception:
                pass

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
            _, hash_value = _getHashValue(attr=attr, eval_context=self._get_eval_context())
            if hash_value:
                attributes[attr] = hash_value
        return attributes

    def refresh_sticky_buckets(self, force: bool = False) -> None:
        if not self.sticky_bucket_service:
            return

        attributes = self._get_sticky_bucket_attributes()
        if not force and attributes == self._sticky_bucket_attributes:
            logger.debug("Skipping refresh of sticky bucket assignments, no changes")
            return

        self._sticky_bucket_attributes = attributes
        self._sticky_bucket_assignment_docs = self.sticky_bucket_service.get_all_assignments(attributes)
        # Update the user context with the new sticky bucket assignment docs
        self._user_ctx.sticky_bucket_assignment_docs = self._sticky_bucket_assignment_docs
