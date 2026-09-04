#!/usr/bin/env python
"""
This is the Python client library for GrowthBook, the open-source
feature flagging and A/B testing platform.
More info at https://www.growthbook.io
"""
import atexit
import json
import threading
import logging
import warnings

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Any, Set, Tuple, List, Dict, Callable, cast

from typing_extensions import deprecated

if TYPE_CHECKING:
    from .plugins.base import PluginLike

from .common_types import (
    T,
    EventLogger,
    FeatureUsageCallback,
    TrackingCallback,
    EvaluationContext,
    Experiment,
    FeatureResult,
    Feature,
    GlobalContext,
    Options,
    Result,
    StackContext,
    UserContext,
    AbstractStickyBucketService,
    AbstractAsyncStickyBucketService,
    FeatureRule,
    build_remote_eval_payload,
    features_from_dict,
    tracking_dedupe_key,
    tracking_user_context,
    validate_remote_eval_options,
)

from base64 import b64decode
from time import time
import aiohttp
import asyncio

from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError, ClientPayloadError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from urllib3 import PoolManager, Timeout

if TYPE_CHECKING:
    # Only present in urllib3 2.x; the runtime dependency allows 1.x too.
    from urllib3.response import BaseHTTPResponse

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
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        pass

    def clear(self) -> None:
        pass


class CacheEntry(object):
    def __init__(self, value: Dict[str, Any], ttl: int) -> None:
        self.value = value
        self.ttl = ttl
        self.expires = time() + ttl

    def update(self, value: Dict[str, Any]) -> None:
        self.value = value
        self.expires = time() + self.ttl


class InMemoryFeatureCache(AbstractFeatureCache):
    def __init__(self) -> None:
        self.cache: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            entry = self.cache[key]
            if entry.expires >= time():
                return entry.value
        return None

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        if key in self.cache:
            self.cache[key].update(value)
        else:
            self.cache[key] = CacheEntry(value, ttl)

    def clear(self) -> None:
        self.cache.clear()

class InMemoryStickyBucketService(AbstractStickyBucketService):
    def __init__(self) -> None:
        self.docs: Dict[str, Dict[str, Any]] = {}

    def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict[str, Any]]:
        return self.docs.get(self.get_key(attributeName, attributeValue), None)

    def save_assignments(self, doc: Dict[str, Any]) -> None:
        self.docs[self.get_key(doc["attributeName"], doc["attributeValue"])] = doc

    def destroy(self) -> None:
        self.docs.clear()


class SSEClient:
    def __init__(
        self,
        api_host: str,
        client_key: str,
        on_event: Callable[[Dict[str, Any]], None],
        reconnect_delay: int = 5,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> None:
        self.api_host = api_host
        self.client_key = client_key

        self.on_event = on_event
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout

        self._sse_session: Optional[aiohttp.ClientSession] = None
        self._sse_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.is_running = False

        self.headers = {
            "Accept": "application/json; q=0.5, text/event-stream",
            "Cache-Control": "no-cache",
            "Accept-Encoding": "gzip, deflate, br",
        }

        if headers:
            self.headers.update(headers)

    def connect(self) -> None:
        if self.is_running:
            logger.debug("Streaming session is already running.")
            return

        self.is_running = True
        self._sse_thread = threading.Thread(target=self._run_sse_channel, daemon=True)
        self._sse_thread.start()
        atexit.register(self.disconnect)

    def disconnect(self, timeout: float = 10) -> None:
        """Gracefully disconnect with timeout"""
        logger.debug("Initiating SSE client disconnect")
        self.is_running = False
        
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._stop_session(timeout), self._loop)
            try:
                # Wait with timeout for clean shutdown
                future.result(timeout=timeout)
                logger.debug("SSE session stopped cleanly")
            except Exception as e:
                logger.warning(f"Error during SSE disconnect: {e}")
                # Force close the loop if clean shutdown failed
                if self._loop and self._loop.is_running():
                    try:
                        self._loop.call_soon_threadsafe(self._loop.stop)
                    except Exception:
                        pass

        if self._sse_thread:
            self._sse_thread.join(timeout=timeout)
            if self._sse_thread.is_alive():
                logger.warning("SSE thread did not terminate gracefully within timeout")
            else:
                logger.debug("SSE thread terminated")

        logger.debug("Streaming session disconnected")

    def _get_sse_url(self, api_host: str, client_key: str) -> str:
        api_host = (api_host or "https://cdn.growthbook.io").rstrip("/")
        return f"{api_host}/sub/{client_key}"

    async def _init_session(self) -> None:
        url = self._get_sse_url(self.api_host, self.client_key)
        
        try:
            while self.is_running:
                try:
                    async with aiohttp.ClientSession(headers=self.headers, 
                        timeout=aiohttp.ClientTimeout(connect=self.timeout)) as session:
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
                    await self._wait_for_reconnect()
                    if not self.is_running:
                        break  # type: ignore[unreachable]
                except TimeoutError:
                    logger.warning(f"Streaming connection timed out after {self.timeout} seconds.")
                    await self._wait_for_reconnect()
                    if not self.is_running:
                        break  # type: ignore[unreachable]
                except asyncio.CancelledError:
                    logger.debug("SSE session cancelled")
                    break
                finally:
                    await self._close_session()
        except asyncio.CancelledError:
            logger.debug("SSE _init_session cancelled")
            pass
        finally:
            # Ensure session is closed on any exit
            await self._close_session()

    async def _process_response(self, response: aiohttp.ClientResponse) -> None:
        event_data: Dict[str, Any] = {}
        try:
            async for line in response.content:
                # Check for cancellation before processing each line
                if not self.is_running:
                    logger.debug("SSE processing stopped - is_running is False")
                    break
                    
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("event:"):
                    event_data['type'] = decoded_line[len("event:"):].strip()
                elif decoded_line.startswith("data:"):
                    # Per W3C EventSource spec, multiple `data:` lines in a
                    # single event are joined with `\n` BETWEEN them, not
                    # prepended to each one. The old logic produced
                    # "\n<line1>\n<line2>" (leading newline), which json.loads
                    # tolerates by luck but breaks for empty-data events
                    # ("\n" alone) and any non-JSON consumer.
                    line_data = decoded_line[len("data:"):].strip()
                    if "data" in event_data:
                        event_data["data"] += "\n" + line_data
                    else:
                        event_data["data"] = line_data
                elif not decoded_line:
                    # End-of-event marker. Per the W3C EventSource spec, an
                    # event with only a `type` (no `data:` line) is still a
                    # valid event — the proxy emits parameter-less
                    # `features-updated` events this way in remote-eval mode as
                    # a cache-invalidation signal. Gating dispatch on
                    # `'data' in event_data` would silently drop them.
                    if 'type' in event_data:
                        try:
                            self.on_event(event_data)
                        except Exception as e:
                            logger.warning(f"Error in event handler: {e}")
                    event_data = {}

            # Process any remaining event data (stream closed without a
            # trailing blank line).
            if 'type' in event_data:
                try:
                    self.on_event(event_data)
                except Exception as e:
                    logger.warning(f"Error in final event handler: {e}")
        except asyncio.CancelledError:
            logger.debug("SSE response processing cancelled")
            raise
        except Exception as e:
            logger.warning(f"Error processing SSE response: {e}")
            raise

    async def _wait_for_reconnect(self) -> None:
        logger.info(f"Attempting to reconnect streaming in {self.reconnect_delay} seconds")
        try:
            await asyncio.sleep(self.reconnect_delay)
        except asyncio.CancelledError:
            logger.debug("Reconnect wait cancelled")
            raise

    async def _close_session(self) -> None:
        if self._sse_session:
            await self._sse_session.close()
            logger.debug("Streaming session closed.")

    def _run_sse_channel(self) -> None:
        self._loop = asyncio.new_event_loop()
        
        try:
            self._loop.run_until_complete(self._init_session())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _stop_session(self, timeout: float = 10) -> None:
        """Stop the SSE session and cancel all tasks with timeout"""
        logger.debug("Stopping SSE session")
        
        # Close the session first
        if self._sse_session and not self._sse_session.closed:
            try:
                await self._sse_session.close()
                logger.debug("SSE session closed")
            except Exception as e:
                logger.warning(f"Error closing SSE session: {e}")

        # Cancel all tasks in this loop
        if self._loop and self._loop.is_running():
            try:
                # Get all tasks for this specific loop
                tasks = [task for task in asyncio.all_tasks(self._loop) 
                        if not task.done() and task is not asyncio.current_task(self._loop)]
                
                if tasks:
                    logger.debug(f"Cancelling {len(tasks)} SSE tasks")
                    # Cancel all tasks
                    for task in tasks:
                        task.cancel()
                    
                    # Wait for tasks to complete with timeout
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=timeout
                        )
                        logger.debug("All SSE tasks cancelled successfully")
                    except asyncio.TimeoutError:
                        logger.warning("Some SSE tasks did not cancel within timeout")
                    except Exception as e:
                        logger.warning(f"Error during task cancellation: {e}")
            except Exception as e:
                logger.warning(f"Error during SSE task cleanup: {e}")

from collections import OrderedDict

# ... (imports)

class FeatureRepository(object):
    def __init__(self) -> None:
        self.cache: AbstractFeatureCache = InMemoryFeatureCache()
        self.http: Optional[PoolManager] = None
        self.http_connect_timeout: Optional[int] = None
        self.http_read_timeout: Optional[int] = None
        self.sse_client: Optional[SSEClient] = None
        self._feature_update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Background refresh support
        self._refresh_thread: Optional[threading.Thread] = None
        self._refresh_stop_event = threading.Event()
        self._refresh_lock = threading.Lock()
        
        # ETag cache for bandwidth optimization
        # Using OrderedDict for LRU cache (max 100 entries)
        self._etag_cache: OrderedDict[str, Tuple[str, Dict[str, Any]]] = OrderedDict()
        self._max_etag_entries = 100
        self._etag_lock = threading.Lock()

    def set_cache(self, cache: AbstractFeatureCache) -> None:
        self.cache = cache

    def clear_cache(self) -> None:
        self.cache.clear()

    def save_in_cache(self, key: str, res: Dict[str, Any], ttl: int = 600) -> None:
        self.cache.set(key, res, ttl)

    def add_feature_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be notified when features are updated due to cache expiry"""
        if callback not in self._feature_update_callbacks:
            self._feature_update_callbacks.append(callback)

    def remove_feature_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a feature update callback"""
        if callback in self._feature_update_callbacks:
            self._feature_update_callbacks.remove(callback)

    def _notify_feature_update_callbacks(self, features_data: Dict[str, Any]) -> None:
        """Notify all registered callbacks about feature updates"""
        for callback in self._feature_update_callbacks:
            try:
                callback(features_data)
            except Exception as e:
                logger.warning(f"Error in feature update callback: {e}")

    # Loads features with an in-memory cache in front using stale-while-revalidate approach
    def load_features(
        self,
        api_host: str,
        client_key: str,
        decryption_key: str = "",
        ttl: int = 600,
        remote_eval: bool = False,
        payload: Optional[Dict[str, Any]] = None,
        cache_key_attributes: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        key = self._compute_cache_key(api_host, client_key, remote_eval, payload, cache_key_attributes)

        # `force_refresh=True` bypasses the cache lookup so SSE invalidation
        # signals (proxy `features-updated`) actually trigger a refetch
        # instead of returning the stale entry.
        cached = None if force_refresh else self.cache.get(key)
        if not cached:
            if remote_eval:
                if payload is None:
                    logger.error("Payload is required for remote-eval POST request")
                    return None
                # Remote-eval responses are not encrypted (server is trusted).
                res = self._fetch_and_decode_post(api_host, client_key, payload)
            else:
                res = self._fetch_features(api_host, client_key, decryption_key)
            if res is not None:
                self.cache.set(key, res, ttl)
                logger.debug("Fetched features from API, stored in cache")
                # Skip global callbacks in remote-eval mode: responses are
                # per-instance/per-user, so broadcasting would cross-pollute
                # other GrowthBook instances sharing this singleton repo.
                if not remote_eval:
                    self._notify_feature_update_callbacks(res)
                return res
        return cached


    async def load_features_async(
        self,
        api_host: str,
        client_key: str,
        decryption_key: str = "",
        ttl: int = 600,
        remote_eval: bool = False,
        payload: Optional[Dict[str, Any]] = None,
        cache_key_attributes: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        key = self._compute_cache_key(api_host, client_key, remote_eval, payload, cache_key_attributes)

        cached = None if force_refresh else self.cache.get(key)
        if not cached:
            if remote_eval:
                if payload is None:
                    logger.error("Payload is required for remote-eval POST request")
                    return None
                res = await self._fetch_and_decode_post_async(api_host, client_key, payload)
            else:
                res = await self._fetch_features_async(api_host, client_key, decryption_key)
            if res is not None:
                self.cache.set(key, res, ttl)
                logger.debug("Fetched features from API, stored in cache")
                if not remote_eval:
                    self._notify_feature_update_callbacks(res)
                return res
        return cached
    
    @property
    def user_agent_suffix(self) -> Optional[str]:
        return getattr(self, "_user_agent_suffix", None)
        
    @user_agent_suffix.setter
    def user_agent_suffix(self, value: Optional[str]) -> None:
        self._user_agent_suffix = value

    # Perform the GET request (separate method for easy mocking)
    def _get(self, url: str, headers: Optional[Dict[str, str]] = None) -> "BaseHTTPResponse":
        timeout = None
        if self.http_connect_timeout and self.http_read_timeout:
            timeout = Timeout(connect=self.http_connect_timeout, read=self.http_read_timeout)
        self.http = self.http or PoolManager(timeout=timeout)
        return self.http.request("GET", url, headers=headers or {})
    
    def _get_headers(self, client_key: str, existing_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = existing_headers or {}
        headers['Accept-Encoding'] = "gzip, deflate"
        
        # Add User-Agent with optional suffix
        ua = "Gb-Python"
        ua += f"-{self.user_agent_suffix}" if self.user_agent_suffix else f"-{client_key[-4:]}"
        headers['User-Agent'] = ua
            
        return headers

    # Perform the POST request (separate method for easy mocking)
    def _post(self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> "BaseHTTPResponse":
        timeout = None
        if self.http_connect_timeout and self.http_read_timeout:
            timeout = Timeout(connect=self.http_connect_timeout, read=self.http_read_timeout)
        self.http = self.http or PoolManager(timeout=timeout)
        body = json.dumps(payload).encode("utf-8")
        return self.http.request("POST", url, body=body, headers=headers or {})

    def _fetch_and_decode_post(
        self, api_host: str, client_key: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        url = self._get_remote_eval_url(api_host, client_key)
        headers = self._get_headers(client_key)
        headers["Content-Type"] = "application/json"
        logger.debug(f"Remote-eval POST to {url}")
        try:
            r = self._post(url, payload, headers)
            if r.status >= 400:
                logger.warning(
                    "Failed to fetch features (remote eval), received status code %d", r.status
                )
                return None
            decoded: Dict[str, Any] = json.loads(r.data.decode("utf-8"))
            return decoded
        except Exception as e:
            logger.warning(f"Failed to decode remote-eval response: {e}")
            return None

    async def _fetch_and_decode_post_async(
        self, api_host: str, client_key: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        url = self._get_remote_eval_url(api_host, client_key)
        headers = self._get_headers(client_key)
        headers["Content-Type"] = "application/json"
        logger.debug(f"[Async] Remote-eval POST to {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        logger.warning(
                            "Failed to fetch features (remote eval), received status code %d",
                            response.status,
                        )
                        return None
                    decoded: Dict[str, Any] = await response.json()
                    return decoded
        except aiohttp.ClientError as e:
            logger.warning(f"HTTP request failed (remote eval): {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to decode remote-eval response: {e}")
            return None

    def _fetch_and_decode(self, api_host: str, client_key: str) -> Optional[Dict[str, Any]]:
        url = self._get_features_url(api_host, client_key)
        headers = self._get_headers(client_key)
        logger.debug(f"Fetching features from {url} with headers {headers}")
        
        # Check if we have a cached ETag for this URL
        cached_etag = None
        cached_data = None
        with self._etag_lock:
            if url in self._etag_cache:
                # Move to end (mark as recently used)
                self._etag_cache.move_to_end(url)
                cached_etag, cached_data = self._etag_cache[url]
                headers['If-None-Match'] = cached_etag
                logger.debug(f"Using cached ETag for request: {cached_etag[:20]}...")
            else:
                logger.debug(f"No ETag cache found for URL: {url}")
        
        try:
            r = self._get(url, headers)
            
            # Handle 304 Not Modified - content hasn't changed
            if r.status == 304:
                logger.debug(f"ETag match! Server returned 304 Not Modified - using cached data (saved bandwidth)")
                if cached_data is not None:
                    logger.debug(f"Returning cached response ({len(str(cached_data))} bytes)")
                    return cached_data
                else:
                    logger.warning("Received 304 but no cached data available")
                    return None
            
            if r.status >= 400:
                logger.warning(
                    "Failed to fetch features, received status code %d", r.status
                )
                return None
            
            decoded: Dict[str, Any] = json.loads(r.data.decode("utf-8"))

            # Store the new ETag if present
            response_etag = r.headers.get('ETag')
            if response_etag:
                with self._etag_lock:
                    self._etag_cache[url] = (response_etag, decoded)
                    # Enforce max size
                    if len(self._etag_cache) > self._max_etag_entries:
                        self._etag_cache.popitem(last=False)
                        
                    if cached_etag:
                        logger.debug(f"ETag updated: {cached_etag[:20]}... -> {response_etag[:20]}...")
                    else:
                        logger.debug(f"New ETag cached: {response_etag[:20]}... ({len(str(decoded))} bytes)")
                    logger.debug(f"ETag cache now contains {len(self._etag_cache)} entries")
            else:
                logger.debug("No ETag header in response")
            
            return decoded
        except Exception as e:
            logger.error(f"Failed to decode feature JSON from GrowthBook API: {e}")
            return None

    async def _fetch_and_decode_async(self, api_host: str, client_key: str) -> Optional[Dict[str, Any]]:
        url = self._get_features_url(api_host, client_key)
        headers = self._get_headers(client_key=client_key)
        logger.debug(f"[Async] Fetching features from {url} with headers {headers}")
        
        # Check if we have a cached ETag for this URL
        cached_etag = None
        cached_data = None
        with self._etag_lock:
            if url in self._etag_cache:
                # Move to end (mark as recently used)
                self._etag_cache.move_to_end(url)
                cached_etag, cached_data = self._etag_cache[url]
                headers['If-None-Match'] = cached_etag
                logger.debug(f"[Async] Using cached ETag for request: {cached_etag[:20]}...")
            else:
                logger.debug(f"[Async] No ETag cache found for URL: {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    # Handle 304 Not Modified - content hasn't changed
                    if response.status == 304:
                        logger.debug(f"[Async] ETag match! Server returned 304 Not Modified - using cached data (saved bandwidth)")
                        if cached_data is not None:
                            logger.debug(f"[Async] Returning cached response ({len(str(cached_data))} bytes)")
                            return cached_data
                        else:
                            logger.warning("[Async] Received 304 but no cached data available")
                            return None
                    
                    if response.status >= 400:
                        logger.warning("Failed to fetch features, received status code %d", response.status)
                        return None
                    
                    decoded: Dict[str, Any] = await response.json()

                    # Store the new ETag if present
                    response_etag = response.headers.get('ETag')
                    if response_etag:
                        with self._etag_lock:
                            self._etag_cache[url] = (response_etag, decoded)
                            # Enforce max size
                            if len(self._etag_cache) > self._max_etag_entries:
                                self._etag_cache.popitem(last=False)
                                
                            if cached_etag:
                                logger.debug(f"[Async] ETag updated: {cached_etag[:20]}... -> {response_etag[:20]}...")
                            else:
                                logger.debug(f"[Async] New ETag cached: {response_etag[:20]}... ({len(str(decoded))} bytes)")
                            logger.debug(f"[Async] ETag cache now contains {len(self._etag_cache)} entries")
                    else:
                        logger.debug("[Async] No ETag header in response")
                    
                    return decoded
        except aiohttp.ClientError as e:
            logger.warning(f"HTTP request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to decode feature JSON from GrowthBook API: {e}")
            return None
        
    def decrypt_response(self, data: Dict[str, Any], decryption_key: str) -> Optional[Dict[str, Any]]:
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
        
        if "encryptedContextualBandits" in data:
            if not decryption_key:
                raise ValueError("Must specify decryption_key")
            try:
                decrypted = decrypt(data["encryptedContextualBandits"], decryption_key)
                data['contextualBandits'] = json.loads(decrypted)
                del data['encryptedContextualBandits']
            except Exception:
                # Drop the undecryptable section (JS decryptPayload deletes the
                # encrypted key either way); absent sections are preserved
                # downstream, so the previous coherent map stays active.
                del data['encryptedContextualBandits']
                logger.warning(
                    "Failed to decrypt contextual bandits from GrowthBook API response"
                )

        if "encryptedSavedGroups" in data:
            if not decryption_key:
                raise ValueError("Must specify decryption_key")
            try:
                decryptedFeatures = decrypt(data["encryptedSavedGroups"], decryption_key)
                data['savedGroups'] = json.loads(decryptedFeatures)
                del data['encryptedSavedGroups']
                return data
            except Exception:
                del data['encryptedSavedGroups']
                logger.warning(
                    "Failed to decrypt saved groups from GrowthBook API response"
                )

        return data

    def decrypt_payload_sections(
        self, payload: Dict[str, Any], decryption_key: str
    ) -> Optional[Dict[str, Any]]:
        """Decrypt any encrypted sections of an SDK payload, returning a copy
        with the plaintext sections in place (JS setPayload accepts encrypted
        payloads the same way). Payloads with no encrypted sections are
        returned as-is; None means the features section failed to decrypt and
        the payload should be discarded."""
        if not any(
            k in payload
            for k in (
                "encryptedFeatures",
                "encryptedContextualBandits",
                "encryptedSavedGroups",
            )
        ):
            return payload
        return self.decrypt_response(dict(payload), decryption_key)

    # Fetch features from the GrowthBook API
    def _fetch_features(
        self, api_host: str, client_key: str, decryption_key: str = ""
    ) -> Optional[Dict[str, Any]]:
        decoded = self._fetch_and_decode(api_host, client_key)
        if not decoded:
            return None

        data = self.decrypt_response(decoded, decryption_key)

        return data

    async def _fetch_features_async(
        self, api_host: str, client_key: str, decryption_key: str = ""
    ) -> Optional[Dict[str, Any]]:
        decoded = await self._fetch_and_decode_async(api_host, client_key)
        if not decoded:
            return None

        data = self.decrypt_response(decoded, decryption_key)

        return data


    def startAutoRefresh(
        self,
        api_host: str,
        client_key: str,
        cb: Callable[[Dict[str, Any]], None],
        streaming_timeout: int = 30,
    ) -> None:
        if not client_key:
            raise ValueError("Must specify `client_key` to start features streaming")
        self.sse_client = self.sse_client or SSEClient(api_host=api_host, client_key=client_key, on_event=cb, timeout=streaming_timeout)
        self.sse_client.connect()

    def stopAutoRefresh(self, timeout: float = 10) -> None:
        """Stop auto refresh with timeout"""
        if self.sse_client:
            self.sse_client.disconnect(timeout=timeout)
            self.sse_client = None
    
    def start_background_refresh(self, api_host: str, client_key: str, decryption_key: str, ttl: int = 600, refresh_interval: int = 300) -> None:
        """Start periodic background refresh task"""

        if not client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        with self._refresh_lock:
            if self._refresh_thread is not None:
                return  # Already running
            
            self._refresh_stop_event.clear()
            self._refresh_thread = threading.Thread(
                target=self._background_refresh_worker,
                args=(api_host, client_key, decryption_key, ttl, refresh_interval),
                daemon=True
            )
            self._refresh_thread.start()
            logger.debug("Started background refresh task")
    
    def _background_refresh_worker(self, api_host: str, client_key: str, decryption_key: str, ttl: int, refresh_interval: int) -> None:
        """Worker method for periodic background refresh"""
        while not self._refresh_stop_event.is_set():
            try:
                # Wait for the refresh interval or stop event
                if self._refresh_stop_event.wait(refresh_interval):
                    break  # Stop event was set
                
                logger.debug("Background refresh for Features - started")
                res = self._fetch_features(api_host, client_key, decryption_key)
                if res is not None:
                    cache_key = api_host + "::" + client_key
                    self.cache.set(cache_key, res, ttl)
                    logger.debug("Background refresh completed")
                    # Notify callbacks about fresh features
                    self._notify_feature_update_callbacks(res)
                else:
                    logger.warning("Background refresh failed")
            except Exception as e:
                logger.warning(f"Background refresh error: {e}")
    
    def stop_background_refresh(self) -> None:
        """Stop background refresh task"""
        self._refresh_stop_event.set()
        
        with self._refresh_lock:
            if self._refresh_thread is not None:
                self._refresh_thread.join(timeout=1.0)  # Wait up to 1 second
                self._refresh_thread = None
                logger.debug("Stopped background refresh task")

    @staticmethod
    def _get_features_url(api_host: str, client_key: str) -> str:
        api_host = (api_host or "https://cdn.growthbook.io").rstrip("/")
        return api_host + "/api/features/" + client_key

    @staticmethod
    def _get_remote_eval_url(api_host: str, client_key: str) -> str:
        api_host = (api_host or "https://cdn.growthbook.io").rstrip("/")
        return api_host + "/api/eval/" + client_key

    @staticmethod
    def _compute_cache_key(
        api_host: str,
        client_key: str,
        remote_eval: bool = False,
        payload: Optional[Dict[str, Any]] = None,
        cache_key_attributes: Optional[List[str]] = None,
    ) -> str:
        base = (api_host or "") + "::" + (client_key or "")
        if not remote_eval or not payload:
            return base
        attrs = payload.get("attributes") or {}
        if cache_key_attributes is not None:
            attrs = {k: attrs[k] for k in cache_key_attributes if k in attrs}
        # forcedFeatures is intentionally excluded from the cache key.
        # Matches the JS SDK: the proxy does not filter on forced features, so
        # responses are identical across forced-feature values. See
        # https://github.com/growthbook/growthbook/blob/main/packages/sdk-js/src/feature-repository.ts (getCacheKey)
        sub = {
            "ca": attrs,
            "fv": payload.get("forcedVariations") or {},
            "url": payload.get("url") or "",
        }
        return base + "||" + json.dumps(sub, sort_keys=True)


# Singleton instance
feature_repo = FeatureRepository()

class GrowthBook(object):
    def __init__(
        self,
        enabled: bool = True,
        attributes: Optional[Dict[str, Any]] = None,
        url: str = "",
        features: Optional[Dict[str, Any]] = None,
        qa_mode: bool = False,
        on_experiment_viewed: Optional[TrackingCallback] = None,
        on_feature_usage: Optional[FeatureUsageCallback] = None,
        api_host: str = "",
        client_key: str = "",
        decryption_key: str = "",
        cache_ttl: int = 600,
        forced_variations: Optional[Dict[str, Any]] = None,
        forced_features: Optional[Dict[str, Any]] = None,
        sticky_bucket_service: Optional[AbstractStickyBucketService] = None,
        sticky_bucket_identifier_attributes: Optional[List[str]] = None,
        saved_groups: Optional[Dict[str, Any]] = None,
        remote_eval: bool = False,
        cache_key_attributes: Optional[List[str]] = None,
        streaming: bool = False,
        streaming_connection_timeout: int = 30,
        stale_while_revalidate: bool = False,
        stale_ttl: int = 300,  # 5 minutes default
        plugins: Optional[List["PluginLike"]] = None,
        skip_all_experiments: bool = False,
        # Deprecated args (camelCase spellings fold into their snake_case
        # equivalents above; the snake_case value wins when both are given)
        trackingCallback: Optional[TrackingCallback] = None,
        qaMode: bool = False,
        user: Optional[Dict[str, Any]] = None,
        groups: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        forcedVariations: Optional[Dict[str, Any]] = None,
        http_connect_timeout: Optional[int] = None,
        http_read_timeout: Optional[int] = None,
        savedGroups: Optional[Dict[str, Any]] = None,
        remoteEval: bool = False,
        cacheKeyAttributes: Optional[List[str]] = None,
        # New in 3.1.0 — appended after ALL 3.0.0 parameters (deprecated ones
        # included) so every existing positional call site keeps its meaning.
        contextual_bandits: Optional[Dict[str, Any]] = None,
        contextualBandits: Optional[Dict[str, Any]] = None,
    ) -> None:
        remote_eval = remote_eval or remoteEval
        saved_groups = saved_groups if saved_groups is not None else savedGroups
        contextual_bandits = contextual_bandits if contextual_bandits is not None else contextualBandits
        cache_key_attributes = cache_key_attributes if cache_key_attributes is not None else cacheKeyAttributes
        self._remoteEval = remote_eval
        self._cacheKeyAttributes = cache_key_attributes

        if isinstance(sticky_bucket_service, AbstractAsyncStickyBucketService):
            raise ValueError(
                "AbstractAsyncStickyBucketService is not supported by the synchronous "
                "GrowthBook class. Use GrowthBookClient, or provide an "
                "AbstractStickyBucketService implementation."
            )

        if remote_eval:
            validate_remote_eval_options(
                client_key, decryption_key, sticky_bucket_service, api_host
            )
            if stale_while_revalidate:
                raise ValueError("stale_while_revalidate is not compatible with remote_eval")

        self._enabled = enabled
        self._attributes = attributes if attributes is not None else {}
        self._url = url
        self._features: Dict[str, Feature] = {}
        self._saved_groups = saved_groups if saved_groups is not None else {}
        self._contextual_bandits = contextual_bandits if contextual_bandits is not None else {}
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl
        self.sticky_bucket_identifier_attributes = sticky_bucket_identifier_attributes
        self.sticky_bucket_service = sticky_bucket_service
        self._sticky_bucket_assignment_docs: Dict[str, Any] = {}
        self._using_derived_sticky_bucket_attributes = not sticky_bucket_identifier_attributes
        self._sticky_bucket_attributes: Optional[Dict[str, Any]] = None

        self._qaMode = qa_mode or qaMode
        if trackingCallback is not None:
            warnings.warn(
                "trackingCallback is deprecated, use on_experiment_viewed instead",
                DeprecationWarning,
            )
        self._trackingCallback: Optional[TrackingCallback] = on_experiment_viewed or trackingCallback
        self._featureUsageCallback: Optional[FeatureUsageCallback] = on_feature_usage
        self._skip_all_experiments = skip_all_experiments

        self._streaming = streaming
        self._streaming_timeout = streaming_connection_timeout
        self._stale_while_revalidate = stale_while_revalidate
        self._stale_ttl = stale_ttl

        # Deprecated args
        self._user = user if user is not None else {}
        self._groups = groups if groups is not None else {}
        self._overrides = overrides if overrides is not None else {}
        self._forcedVariations = forced_variations if forced_variations is not None else (forcedVariations if forcedVariations is not None else {})
        self._forcedFeatures: Dict[str, Any] = forced_features or {}

        self._tracked: Dict[str, Any] = {}
        self._assigned: Dict[str, Any] = {}
        self._subscriptions: Set[Callable[[Experiment[Any], Result[Any]], None]] = set()
        self._is_updating_features = False
        # Serializes payload writers (set_features/set_payload/refreshes).
        # Re-entrant because _ingest_payload calls set_features while holding
        # it. Evals stay lock-free — they read the published snapshot.
        self._payload_lock = threading.RLock()
        self._event_logger: Optional[EventLogger] = None

        # support plugins
        self._plugins: List["PluginLike"] = plugins if plugins is not None else []
        self._initialized_plugins: List["PluginLike"] = []

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
            saved_groups=self._saved_groups,
            contextual_bandits=self._contextual_bandits
        )
        # Create a user context for the current user
        self._user_ctx: UserContext = UserContext(
            url=self._url,
            attributes=self._attributes,
            groups=self._groups,
            forced_variations=self._forcedVariations,
            forced_features=self._forcedFeatures,
            overrides=self._overrides,
            sticky_bucket_assignment_docs=self._sticky_bucket_assignment_docs,
            skip_all_experiments=self._skip_all_experiments
        )

        if features:
            self.set_features(features)

        # Register for automatic feature updates when cache expires.
        # Skip in remote-eval mode: responses are per-instance, so the global
        # callback would cross-pollute other GrowthBook instances sharing the
        # singleton FeatureRepository.
        if self._client_key and not self._remoteEval:
            feature_repo.add_feature_update_callback(self._on_feature_update)

        self._initialize_plugins()

        if self._streaming:
            self.load_features()
            self.start_auto_refresh()
        elif self._stale_while_revalidate:
            # Start background refresh task for stale-while-revalidate
            self.load_features()  # Initial load
            feature_repo.start_background_refresh(
                self._api_host, self._client_key, self._decryption_key,
                self._cache_ttl, self._stale_ttl
            )
        elif self._remoteEval:
            # Initial POST to /api/eval/{client_key} so features are populated
            # before the first eval. Matches the JS SDK init() behavior.
            self.load_features()

        if http_connect_timeout and http_read_timeout:
            feature_repo.http_connect_timeout = http_connect_timeout
            feature_repo.http_read_timeout = http_read_timeout

    def _remote_eval_payload(self) -> Dict[str, Any]:
        return build_remote_eval_payload(
            self._attributes, self._forcedVariations, self._url,
            forced_features=self._forcedFeatures,
        )

    def _on_feature_update(self, features_data: Dict[str, Any]) -> None:
        """Callback to handle automatic feature updates from FeatureRepository"""
        if features_data:
            self._ingest_payload(features_data)

    def _ingest_payload(self, data: Dict[str, Any]) -> None:
        """Apply the sections present in a (decrypted) SDK payload.

        Sections absent from the payload are preserved (JS setPayload
        semantics), and the evaluation context is republished even for
        map-only payloads so a savedGroups/contextualBandits update takes
        effect without waiting for the next features update.

        Writers are serialized: without the lock, two concurrent updates
        (e.g. set_payload and a background refresh) could interleave their
        section writes and publish a snapshot mixing payload generations."""
        with self._payload_lock:
            if "savedGroups" in data:
                self._saved_groups = data["savedGroups"]
            if "contextualBandits" in data:
                self._contextual_bandits = data["contextualBandits"]
            if "features" in data:
                self.set_features(data["features"])
            elif "savedGroups" in data or "contextualBandits" in data:
                self._publish_global_context()

    def _publish_global_context(self) -> None:
        # Swap in a complete snapshot with a single reference rebind
        # (atomic under the GIL) so concurrent lock-free evals never observe
        # features from one payload generation combined with savedGroups or
        # contextualBandits from another. In-flight evals keep the previous
        # coherent snapshot; the async client works the same way.
        self._global_ctx = replace(
            self._global_ctx,
            features=self._features,
            saved_groups=self._saved_groups,
            contextual_bandits=self._contextual_bandits,
        )

    def set_payload(self, payload: Dict[str, Any]) -> None:
        """Set features, saved groups, and contextual bandits from a full SDK
        payload, e.g. one fetched out-of-band from the GrowthBook API.
        Mirrors the JS SDK's setPayload: only the sections present in the
        payload are overwritten, and encrypted sections are decrypted with
        the configured decryption_key."""
        data = feature_repo.decrypt_payload_sections(payload, self._decryption_key)
        if data is not None:
            self._on_feature_update(data)

    def load_features(self, force_refresh: bool = False) -> None:
        """Load features from the configured endpoint, populating the cache.

        `force_refresh=True` bypasses the in-memory cache to honor a fresh
        signal from the proxy (e.g., an SSE `features-updated` event).
        Without it, an immediate `load_features()` after such a signal
        would just return the stale cached payload — defeating the
        invalidation."""
        payload = self._remote_eval_payload() if self._remoteEval else None
        response = feature_repo.load_features(
            self._api_host,
            self._client_key,
            self._decryption_key,
            self._cache_ttl,
            remote_eval=self._remoteEval,
            payload=payload,
            cache_key_attributes=self._cacheKeyAttributes,
            force_refresh=force_refresh,
        )
        if response is not None:
            self._ingest_payload(response)

    async def load_features_async(self, force_refresh: bool = False) -> None:
        if not self._client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        payload = self._remote_eval_payload() if self._remoteEval else None
        features = await feature_repo.load_features_async(
            self._api_host,
            self._client_key,
            self._decryption_key,
            self._cache_ttl,
            remote_eval=self._remoteEval,
            payload=payload,
            cache_key_attributes=self._cacheKeyAttributes,
            force_refresh=force_refresh,
        )

        if features is not None:
            self._ingest_payload(features)

    def _features_event_handler(self, features: str) -> None:
        decoded = json.loads(features)
        if not decoded:
            return None
        
        data = feature_repo.decrypt_response(decoded, self._decryption_key)
        key = self._api_host + "::" + self._client_key

        if data is not None:
            self._ingest_payload(data)
            feature_repo.save_in_cache(key, data, self._cache_ttl)

    def _dispatch_sse_event(self, event_data: Dict[str, Any]) -> None:
        event_type = event_data.get('type')
        if event_type == 'features-updated':
            # In remote-eval mode the proxy emits this event with no inline
            # payload (the payload would be per-user). load_features() handles
            # both modes. force_refresh=True is essential — without it the
            # cache hit would return the stale payload and the invalidation
            # signal would be silently dropped.
            self.load_features(force_refresh=True)
        elif event_type == 'features':
            if self._remoteEval:
                # Defensive: proxy shouldn't send inline payloads to remote-eval
                # clients, but if one arrives, ignore it (not user-filtered) and
                # refetch via the remote-eval path. force_refresh for the same
                # reason as above.
                self.load_features(force_refresh=True)
            else:
                self._features_event_handler(event_data.get('data', '{}'))


    def start_auto_refresh(self) -> None:
        if not self._client_key:
            raise ValueError("Must specify `client_key` to start features streaming")

        feature_repo.startAutoRefresh(
            api_host=self._api_host,
            client_key=self._client_key,
            cb=self._dispatch_sse_event,
            streaming_timeout=self._streaming_timeout
        )

    @deprecated("startAutoRefresh is deprecated, use start_auto_refresh instead")
    def startAutoRefresh(self) -> None:
        return self.start_auto_refresh()

    def stop_auto_refresh(self, timeout: float = 10) -> None:
        """Stop auto refresh with timeout"""
        try:
            if hasattr(feature_repo, 'sse_client') and feature_repo.sse_client:
                feature_repo.sse_client.disconnect(timeout=timeout)
            else:
                feature_repo.stopAutoRefresh()
        except Exception as e:
            logger.warning(f"Error stopping auto refresh: {e}")

    @deprecated("stopAutoRefresh is deprecated, use stop_auto_refresh instead")
    def stopAutoRefresh(self, timeout: float = 10) -> None:
        return self.stop_auto_refresh(timeout=timeout)

    @deprecated("setFeatures is deprecated, use set_features instead")
    def setFeatures(self, features: Dict[str, Any]) -> None:
        return self.set_features(features)

    def set_features(self, features: Dict[str, Any]) -> None:
        # Prevent infinite recursion during feature updates
        self._is_updating_features = True
        try:
            with self._payload_lock:
                self._features = {}
                for key, feature in features.items():
                    if isinstance(feature, Feature):
                        self._features[key] = feature
                    else:
                        self._features[key] = Feature(
                            rules=feature.get("rules", []),
                            defaultValue=feature.get("defaultValue", None),
                        )
                self._publish_global_context()
                self.refresh_sticky_buckets()
        finally:
            self._is_updating_features = False

    @deprecated("getFeatures is deprecated, use get_features instead")
    def getFeatures(self) -> Dict[str, Feature]:
        return self.get_features()

    def get_features(self) -> Dict[str, Feature]:
        return self._features

    @deprecated("setAttributes is deprecated, use set_attributes instead")
    def setAttributes(self, attributes: Dict[str, Any]) -> None:
        return self.set_attributes(attributes)

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        self._attributes = attributes
        self.refresh_sticky_buckets()
        if self._remoteEval and self._client_key:
            # Blocking refetch — matches JS SDK semantics. Known cost of remote
            # eval: every set_attributes call hits the network.
            self.load_features()

    def set_forced_variations(self, forced_variations: Dict[str, Any]) -> None:
        self._forcedVariations = forced_variations or {}
        if self._user_ctx is not None:
            self._user_ctx.forced_variations = self._forcedVariations
        if self._remoteEval and self._client_key:
            self.load_features()

    def set_forced_features(self, forced_features: Dict[str, Any]) -> None:
        """Set forced feature values. The proxy server uses them to filter the
        response in remote-eval mode; local evaluation does NOT consult them
        today (matches the JS SDK behavior). Triggers a refetch when
        remote_eval is enabled."""
        self._forcedFeatures = forced_features or {}
        if self._user_ctx is not None:
            self._user_ctx.forced_features = self._forcedFeatures
        if self._remoteEval and self._client_key:
            self.load_features()

    def set_url(self, url: str) -> None:
        self._url = url or ""
        if self._user_ctx is not None:
            self._user_ctx.url = self._url
        if self._remoteEval and self._client_key:
            self.load_features()

    @deprecated("getAttributes is deprecated, use get_attributes instead")
    def getAttributes(self) -> Dict[str, Any]:
        return self.get_attributes()

    def get_attributes(self) -> Dict[str, Any]:
        return self._attributes

    def destroy(self, timeout: float = 10) -> None:
        """Gracefully destroy the GrowthBook instance"""
        logger.debug("Starting GrowthBook destroy process")
        
        try:
            # Clean up plugins
            logger.debug("Cleaning up plugins")
            self._cleanup_plugins()
        except Exception as e:
            logger.warning(f"Error cleaning up plugins: {e}")
        
        try:
            logger.debug("Stopping auto refresh during destroy")
            self.stop_auto_refresh(timeout=timeout)
        except Exception as e:
            logger.warning(f"Error stopping auto refresh during destroy: {e}")
        
        try:
            # Stop background refresh operations
            if self._stale_while_revalidate and self._client_key:
                feature_repo.stop_background_refresh()
        except Exception as e:
            logger.warning(f"Error stopping background refresh during destroy: {e}")
        
        try:
            # Clean up feature update callback (not registered in remote-eval mode)
            if self._client_key and not self._remoteEval:
                feature_repo.remove_feature_update_callback(self._on_feature_update)
        except Exception as e:
            logger.warning(f"Error removing feature update callback: {e}")
        
        # Clear all internal state
        try:
            self._subscriptions.clear()
            self._tracked.clear()
            self._assigned.clear()
            self._trackingCallback = None
            self._featureUsageCallback = None
            self._event_logger = None
            self._forcedVariations.clear()
            self._overrides.clear()
            self._groups.clear()
            self._attributes.clear()
            self._features.clear()
            logger.debug("GrowthBook instance destroyed successfully")
        except Exception as e:
            logger.warning(f"Error clearing internal state: {e}")

    def set_event_logger(self, fn: EventLogger) -> None:
        """Register a callable that will be invoked by log_event.

        The callable receives (event_name: str, properties: dict, user_context: UserContext).
        Typically set by GrowthBookTrackingPlugin rather than called directly.
        """
        self._event_logger = fn

    def log_event(self, event_name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Log a custom event to the GrowthBook ingestor.

        Requires GrowthBookTrackingPlugin to be configured; without it a warning
        is emitted and the call is a no-op.

        Args:
            event_name: Name of the event (e.g. ``"button_clicked"``).
            properties: Optional dict of event-specific properties.
        """
        if self._event_logger is None:
            logger.warning(
                "log_event called but no event logger is configured. "
                "Add GrowthBookTrackingPlugin to enable event logging."
            )
            return
        # Same sync the eval path does — otherwise the event_logger callback
        # receives a UserContext with stale forced_variations / forced_features /
        # overrides (this used to be a separate two-field manual sync that
        # missed those — see _sync_user_ctx_from_instance for the rationale).
        self._sync_user_ctx_from_instance()
        try:
            self._event_logger(event_name, properties or {}, self._user_ctx)
        except Exception as e:
            logger.exception("Error in event logger: %s", e)

    @deprecated("isOn is deprecated, use is_on instead")
    def isOn(self, key: str) -> bool:
        return self.is_on(key)

    def is_on(self, key: str) -> bool:
        return self.eval_feature(key).on

    @deprecated("isOff is deprecated, use is_off instead")
    def isOff(self, key: str) -> bool:
        return self.is_off(key)

    def is_off(self, key: str) -> bool:
        return self.eval_feature(key).off

    @deprecated("getFeatureValue is deprecated, use get_feature_value instead")
    def getFeatureValue(self, key: str, fallback: T) -> T:
        return self.get_feature_value(key, fallback)

    def get_feature_value(self, key: str, fallback: T) -> T:
        res = self.eval_feature(key)
        return cast(T, res.value) if res.value is not None else fallback

    @deprecated("evalFeature is deprecated, use eval_feature instead")
    def evalFeature(self, key: str) -> FeatureResult[Any]:
        return self.eval_feature(key)
    
    def _ensure_fresh_features(self) -> None:
        """Lazy refresh: Check cache expiry and refresh if needed, but only if client_key is provided"""
        
        # Prevent infinite recursion when updating features (e.g., during sticky bucket refresh)
        if self._is_updating_features:
            return
        
        if self._streaming or self._stale_while_revalidate or not self._client_key:
            return  # Skip cache checks - SSE or background refresh handles freshness

        try:
            self.load_features()
        except Exception as e:
            logger.warning(f"Failed to refresh features: {e}")

    def _sync_user_ctx_from_instance(self) -> None:
        """Single source of truth for instance state → `_user_ctx` propagation.

        Every code path that hands `_user_ctx` to a caller-facing callback
        (`_get_eval_context`, `log_event`, anywhere else in the future) MUST
        call this first. Otherwise direct mutations like
        `gb._attributes["foo"] = "bar"` — or even a missed setter sync — leave
        the user_context the callback sees in a stale, inconsistent state.

        """
        self._user_ctx.attributes = self._attributes
        self._user_ctx.url = self._url
        self._user_ctx.overrides = self._overrides
        self._user_ctx.forced_variations = self._forcedVariations
        self._user_ctx.forced_features = self._forcedFeatures
        # NOTE: sticky_bucket_assignment_docs has its own refresh flow via
        # refresh_sticky_buckets(); intentionally NOT mirrored here. `groups`
        # and `skip_all_experiments` have no setters today so they don't drift.

    def _build_eval_context(self) -> EvaluationContext:
        """Assemble an EvaluationContext WITHOUT any side effects.

        Unlike `_get_eval_context`, this never triggers a feature refresh, so
        it is safe to call from inside the feature-refresh / sticky-bucket flow
        (see `_get_sticky_bucket_attributes`). Calling `_get_eval_context` there
        would re-enter `_ensure_fresh_features` -> `load_features` for every
        sticky-bucket identifier attribute, causing redundant feature reloads.
        """
        # Centralized sync (see _sync_user_ctx_from_instance for rationale).
        self._sync_user_ctx_from_instance()
        # global_ctx.options.url is not part of _user_ctx; still needs updating.
        self._global_ctx.options.url = self._url
        return EvaluationContext(
            global_ctx = self._global_ctx,
            user = self._user_ctx,
            stack = StackContext(evaluated_features=set()),
            # Wired only when a consumer exists (contexts are per-eval, so a
            # callback installed later — e.g. by a plugin — is still picked
            # up), letting core skip dead work like rule.tracks hydration.
            tracking_cb = self._track if self._trackingCallback else None,
            callback_subscription = self._fireSubscriptions,
            feature_usage_cb = self._feature_usage if self._featureUsageCallback else None,
        )

    def _get_eval_context(self) -> EvaluationContext:
        # Lazy refresh: ensure features are fresh before evaluation, then build
        # the (side-effect-free) context.
        self._ensure_fresh_features()
        return self._build_eval_context()

    def eval_feature(self, key: str) -> FeatureResult[Any]:
        return core_eval_feature(key=key, evalContext=self._get_eval_context())

    def _feature_usage(self, key: str, result: FeatureResult[Any], user_context: UserContext) -> None:
        if not self._featureUsageCallback:
            return
        try:
            # Snapshot so the logged attributes are exactly the ones used
            # for the evaluation, even if the caller mutates them afterwards.
            self._featureUsageCallback(key, result, tracking_user_context(user_context))
        except Exception:
            pass

    @deprecated("getAllResults is deprecated, use get_all_results instead")
    def getAllResults(self) -> Dict[str, Dict[str, Any]]:
        return self.get_all_results()

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        return self._assigned.copy()

    def _fireSubscriptions(self, experiment: Experiment[Any], result: Result[Any]) -> None:
        if experiment is not None:
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

    def run(self, experiment: Experiment[T]) -> Result[T]:
        result = run_experiment(experiment=experiment,
                                evalContext=self._get_eval_context())

        self._fireSubscriptions(experiment, result)
        return result

    def subscribe(self, callback: Callable[[Experiment[Any], Result[Any]], None]) -> Callable[[], None]:
        self._subscriptions.add(callback)
        return lambda: self._subscriptions.remove(callback)

    def _track(self, experiment: Experiment[Any], result: Result[Any], user_context: UserContext) -> None:
        if not self._trackingCallback:
            return None
        key = tracking_dedupe_key(experiment, result)
        if not self._tracked.get(key):
            try:
                # Snapshot so the logged attributes are exactly the ones used
                # for bucketing, even if the caller mutates them afterwards.
                self._trackingCallback(
                    experiment=experiment,
                    result=result,
                    user_context=tracking_user_context(user_context),
                )
                self._tracked[key] = True
            except Exception as e:
                logger.exception(e)

    def _derive_sticky_bucket_identifier_attributes(self) -> List[str]:
        attributes = set()
        for key, feature in self._features.items():
            for rule in feature.rules:
                if rule.variations or rule.contextualVariations:
                    attributes.add(rule.hashAttribute or "id")
                    if rule.fallbackAttribute:
                        attributes.add(rule.fallbackAttribute)
        return list(attributes)

    def _get_sticky_bucket_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, str] = {}
        if self._using_derived_sticky_bucket_attributes:
            self.sticky_bucket_identifier_attributes = self._derive_sticky_bucket_identifier_attributes()

        if not self.sticky_bucket_identifier_attributes:
            return attributes

        # Build the context once, side-effect-free. Using _get_eval_context()
        # here would re-trigger _ensure_fresh_features() -> load_features() on
        # every attribute (this method itself runs inside the refresh flow).
        eval_context = self._build_eval_context()
        for attr in self.sticky_bucket_identifier_attributes:
            _, hash_value = _getHashValue(attr=attr, eval_context=eval_context)
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

    def _initialize_plugins(self) -> None:
        """Initialize all plugins with this GrowthBook instance."""
        for plugin in self._plugins:
            try:
                # getattr (not hasattr+access) keeps duck-typed plugin objects
                # working while narrowing cleanly under both checkers.
                initialize = getattr(plugin, "initialize", None)
                if callable(initialize):
                    # Plugin is a class instance with initialize method
                    initialize(self)
                    self._initialized_plugins.append(plugin)
                    logger.debug(f"Initialized plugin: {plugin.__class__.__name__}")
                elif callable(plugin):
                    # Plugin is a callable function
                    plugin(self)
                    self._initialized_plugins.append(plugin)
                    logger.debug(f"Initialized callable plugin: {getattr(plugin, '__name__', plugin)}")
                else:
                    logger.warning(f"Plugin {plugin} is neither callable nor has initialize method")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin}: {e}")

    @property
    def user_agent_suffix(self) -> Optional[str]:
        """Get the suffix appended to the User-Agent header"""
        return feature_repo.user_agent_suffix
        
    @user_agent_suffix.setter
    def user_agent_suffix(self, value: Optional[str]) -> None:
        """Set a suffix to be appended to the User-Agent header"""
        feature_repo.user_agent_suffix = value

    def _cleanup_plugins(self) -> None:
        """Cleanup all initialized plugins."""
        for plugin in self._initialized_plugins:
            try:
                cleanup = getattr(plugin, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                    logger.debug(f"Cleaned up plugin: {plugin.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin}: {e}")
        self._initialized_plugins.clear()
