#!/usr/bin/env python
"""
This is the Python client library for GrowthBook, the open-source
feature flagging and A/B testing platform.
More info at https://www.growthbook.io
"""

import re
import sys
import json
from abc import ABC, abstractmethod
import threading
import logging

from typing import Optional, Any, Set, Tuple, List, Dict

# Only require typing_extensions if using Python 3.7 or earlier
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from urllib.parse import urlparse, parse_qs
from base64 import b64decode
from time import time
import aiohttp
import asyncio

from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError, ClientPayloadError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from urllib3 import PoolManager

logger = logging.getLogger("growthbook")


def fnv1a32(str: str) -> int:
    hval = 0x811C9DC5
    prime = 0x01000193
    uint32_max = 2 ** 32
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * prime) % uint32_max
    return hval


def gbhash(seed: str, value: str, version: int) -> Optional[float]:
    if version == 2:
        n = fnv1a32(str(fnv1a32(seed + value)))
        return (n % 10000) / 10000
    if version == 1:
        n = fnv1a32(value + seed)
        return (n % 1000) / 1000
    return None


def inRange(n: float, range: Tuple[float, float]) -> bool:
    return range[0] <= n < range[1]


def inNamespace(userId: str, namespace: Tuple[str, float, float]) -> bool:
    n = gbhash("__" + namespace[0], userId, 1)
    if n is None:
        return False
    return namespace[1] <= n < namespace[2]


def getEqualWeights(numVariations: int) -> List[float]:
    if numVariations < 1:
        return []
    return [1 / numVariations for _ in range(numVariations)]


def getBucketRanges(
    numVariations: int, coverage: float = 1, weights: List[float] = None
) -> List[Tuple[float, float]]:
    if coverage < 0:
        coverage = 0
    if coverage > 1:
        coverage = 1
    if weights is None:
        weights = getEqualWeights(numVariations)
    if len(weights) != numVariations:
        weights = getEqualWeights(numVariations)
    if sum(weights) < 0.99 or sum(weights) > 1.01:
        weights = getEqualWeights(numVariations)

    cumulative: float = 0
    ranges = []
    for w in weights:
        start = cumulative
        cumulative += w
        ranges.append((start, start + coverage * w))

    return ranges


def chooseVariation(n: float, ranges: List[Tuple[float, float]]) -> int:
    for i, r in enumerate(ranges):
        if inRange(n, r):
            return i
    return -1


def getQueryStringOverride(id: str, url: str, numVariations: int) -> Optional[int]:
    res = urlparse(url)
    if not res.query:
        return None
    qs = parse_qs(res.query)
    if id not in qs:
        return None
    variation = qs[id][0]
    if variation is None or not variation.isdigit():
        return None
    varId = int(variation)
    if varId < 0 or varId >= numVariations:
        return None
    return varId


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


def paddedVersionString(input) -> str:
    # If input is a number, convert to a string
    if type(input) is int or type(input) is float:
        input = str(input)

    if not input or type(input) is not str:
        input = "0"

    # Remove build info and leading `v` if any
    input = re.sub(r"(^v|\+.*$)", "", input)
    # Split version into parts (both core version numbers and pre-release tags)
    # "v1.2.3-rc.1+build123" -> ["1","2","3","rc","1"]
    parts = re.split(r"[-.]", input)
    # If it's SemVer without a pre-release, add `~` to the end
    # ["1","0","0"] -> ["1","0","0","~"]
    # "~" is the largest ASCII character, so this will make "1.0.0" greater than "1.0.0-beta" for example
    if len(parts) == 3:
        parts.append("~")
    # Left pad each numeric part with spaces so string comparisons will work ("9">"10", but " 9"<"10")
    # Then, join back together into a single string
    return "-".join([v.rjust(5, " ") if re.match(r"^[0-9]+$", v) else v for v in parts])


def isIn(conditionValue, attributeValue) -> bool:
    if type(attributeValue) is list:
        return bool(set(conditionValue) & set(attributeValue))
    return attributeValue in conditionValue


def evalCondition(attributes: dict, condition: dict, savedGroups: dict = None) -> bool:
    if "$or" in condition:
        return evalOr(attributes, condition["$or"], savedGroups)
    if "$nor" in condition:
        return not evalOr(attributes, condition["$nor"], savedGroups)
    if "$and" in condition:
        return evalAnd(attributes, condition["$and"], savedGroups)
    if "$not" in condition:
        return not evalCondition(attributes, condition["$not"], savedGroups)

    for key, value in condition.items():
        if not evalConditionValue(value, getPath(attributes, key), savedGroups):
            return False

    return True


def evalOr(attributes, conditions, savedGroups) -> bool:
    if len(conditions) == 0:
        return True

    for condition in conditions:
        if evalCondition(attributes, condition, savedGroups):
            return True
    return False


def evalAnd(attributes, conditions, savedGroups) -> bool:
    for condition in conditions:
        if not evalCondition(attributes, condition, savedGroups):
            return False
    return True


def isOperatorObject(obj) -> bool:
    for key in obj.keys():
        if key[0] != "$":
            return False
    return True


def getType(attributeValue) -> str:
    t = type(attributeValue)

    if attributeValue is None:
        return "null"
    if t is int or t is float:
        return "number"
    if t is str:
        return "string"
    if t is list or t is set:
        return "array"
    if t is dict:
        return "object"
    if t is bool:
        return "boolean"
    return "unknown"


def getPath(attributes, path):
    current = attributes
    for segment in path.split("."):
        if type(current) is dict and segment in current:
            current = current[segment]
        else:
            return None
    return current


def evalConditionValue(conditionValue, attributeValue, savedGroups) -> bool:
    if type(conditionValue) is dict and isOperatorObject(conditionValue):
        for key, value in conditionValue.items():
            if not evalOperatorCondition(key, attributeValue, value, savedGroups):
                return False
        return True
    return conditionValue == attributeValue


def elemMatch(condition, attributeValue, savedGroups) -> bool:
    if not type(attributeValue) is list:
        return False

    for item in attributeValue:
        if isOperatorObject(condition):
            if evalConditionValue(condition, item, savedGroups):
                return True
        else:
            if evalCondition(item, condition, savedGroups):
                return True

    return False


def compare(val1, val2) -> int:
    if (type(val1) is int or type(val1) is float) and not (type(val2) is int or type(val2) is float):
        if (val2 is None):
            val2 = 0
        else:
            val2 = float(val2)

    if (type(val2) is int or type(val2) is float) and not (type(val1) is int or type(val1) is float):
        if (val1 is None):
            val1 = 0
        else:
            val1 = float(val1)

    if val1 > val2:
        return 1
    if val1 < val2:
        return -1
    return 0


def evalOperatorCondition(operator, attributeValue, conditionValue, savedGroups) -> bool:
    if operator == "$eq":
        try:
            return compare(attributeValue, conditionValue) == 0
        except Exception:
            return False
    elif operator == "$ne":
        try:
            return compare(attributeValue, conditionValue) != 0
        except Exception:
            return False
    elif operator == "$lt":
        try:
            return compare(attributeValue, conditionValue) < 0
        except Exception:
            return False
    elif operator == "$lte":
        try:
            return compare(attributeValue, conditionValue) <= 0
        except Exception:
            return False
    elif operator == "$gt":
        try:
            return compare(attributeValue, conditionValue) > 0
        except Exception:
            return False
    elif operator == "$gte":
        try:
            return compare(attributeValue, conditionValue) >= 0
        except Exception:
            return False
    elif operator == "$veq":
        return paddedVersionString(attributeValue) == paddedVersionString(conditionValue)
    elif operator == "$vne":
        return paddedVersionString(attributeValue) != paddedVersionString(conditionValue)
    elif operator == "$vlt":
        return paddedVersionString(attributeValue) < paddedVersionString(conditionValue)
    elif operator == "$vlte":
        return paddedVersionString(attributeValue) <= paddedVersionString(conditionValue)
    elif operator == "$vgt":
        return paddedVersionString(attributeValue) > paddedVersionString(conditionValue)
    elif operator == "$vgte":
        return paddedVersionString(attributeValue) >= paddedVersionString(conditionValue)
    elif operator == "$inGroup":
        if not type(conditionValue) is str:
            return False
        if not conditionValue in savedGroups:
            return False
        return isIn(savedGroups[conditionValue] or [], attributeValue)
    elif operator == "$notInGroup":
        if not type(conditionValue) is str:
            return False
        if not conditionValue in savedGroups:
            return True
        return not isIn(savedGroups[conditionValue] or [], attributeValue)
    elif operator == "$regex":
        try:
            r = re.compile(conditionValue)
            return bool(r.search(attributeValue))
        except Exception:
            return False
    elif operator == "$in":
        if not type(conditionValue) is list:
            return False
        return isIn(conditionValue, attributeValue)
    elif operator == "$nin":
        if not type(conditionValue) is list:
            return False
        return not isIn(conditionValue, attributeValue)
    elif operator == "$elemMatch":
        return elemMatch(conditionValue, attributeValue, savedGroups)
    elif operator == "$size":
        if not (type(attributeValue) is list):
            return False
        return evalConditionValue(conditionValue, len(attributeValue), savedGroups)
    elif operator == "$all":
        if not (type(attributeValue) is list):
            return False
        for cond in conditionValue:
            passing = False
            for attr in attributeValue:
                if evalConditionValue(cond, attr, savedGroups):
                    passing = True
            if not passing:
                return False
        return True
    elif operator == "$exists":
        if not conditionValue:
            return attributeValue is None
        return attributeValue is not None
    elif operator == "$type":
        return getType(attributeValue) == conditionValue
    elif operator == "$not":
        return not evalConditionValue(conditionValue, attributeValue, savedGroups)
    return False


class VariationMeta(TypedDict):
    key: str
    name: str
    passthrough: bool


class Filter(TypedDict):
    seed: str
    ranges: List[Tuple[float, float]]
    hashVersion: int
    attribute: str


class Experiment(object):
    def __init__(
        self,
        key: str,
        variations: list,
        weights: List[float] = None,
        active: bool = True,
        status: str = "running",
        coverage: int = None,
        condition: dict = None,
        namespace: Tuple[str, float, float] = None,
        url: str = "",
        include=None,
        groups: list = None,
        force: int = None,
        hashAttribute: str = "id",
        fallbackAttribute: str = None,
        hashVersion: int = None,
        ranges: List[Tuple[float, float]] = None,
        meta: List[VariationMeta] = None,
        filters: List[Filter] = None,
        seed: str = None,
        name: str = None,
        phase: str = None,
        disableStickyBucketing: bool = False,
        bucketVersion: int = None,
        minBucketVersion: int = None,
        parentConditions: List[dict] = None,
    ) -> None:
        self.key = key
        self.variations = variations
        self.weights = weights
        self.active = active
        self.coverage = coverage
        self.condition = condition
        self.namespace = namespace
        self.force = force
        self.hashAttribute = hashAttribute
        self.hashVersion = hashVersion or 1
        self.ranges = ranges
        self.meta = meta
        self.filters = filters
        self.seed = seed
        self.name = name
        self.phase = phase
        self.disableStickyBucketing = disableStickyBucketing
        self.bucketVersion = bucketVersion or 0
        self.minBucketVersion = minBucketVersion or 0
        self.parentConditions = parentConditions

        self.fallbackAttribute = None
        if not self.disableStickyBucketing:
            self.fallbackAttribute = fallbackAttribute

        # Deprecated properties
        self.status = status
        self.url = url
        self.include = include
        self.groups = groups

    def to_dict(self):
        obj = {
            "key": self.key,
            "variations": self.variations,
            "weights": self.weights,
            "active": self.active,
            "coverage": self.coverage or 1,
            "condition": self.condition,
            "namespace": self.namespace,
            "force": self.force,
            "hashAttribute": self.hashAttribute,
            "hashVersion": self.hashVersion,
            "ranges": self.ranges,
            "meta": self.meta,
            "filters": self.filters,
            "seed": self.seed,
            "name": self.name,
            "phase": self.phase,
        }

        if self.fallbackAttribute:
            obj["fallbackAttribute"] = self.fallbackAttribute
        if self.disableStickyBucketing:
            obj["disableStickyBucketing"] = True
        if self.bucketVersion:
            obj["bucketVersion"] = self.bucketVersion
        if self.minBucketVersion:
            obj["minBucketVersion"] = self.minBucketVersion
        if self.parentConditions:
            obj["parentConditions"] = self.parentConditions

        return obj

    def update(self, data: dict) -> None:
        weights = data.get("weights", None)
        status = data.get("status", None)
        coverage = data.get("coverage", None)
        url = data.get("url", None)
        groups = data.get("groups", None)
        force = data.get("force", None)

        if weights is not None:
            self.weights = weights
        if status is not None:
            self.status = status
        if coverage is not None:
            self.coverage = coverage
        if url is not None:
            self.url = url
        if groups is not None:
            self.groups = groups
        if force is not None:
            self.force = force


class Result(object):
    def __init__(
        self,
        variationId: int,
        inExperiment: bool,
        value,
        hashUsed: bool,
        hashAttribute: str,
        hashValue: str,
        featureId: Optional[str],
        meta: VariationMeta = None,
        bucket: float = None,
        stickyBucketUsed: bool = False,
    ) -> None:
        self.variationId = variationId
        self.inExperiment = inExperiment
        self.value = value
        self.hashUsed = hashUsed
        self.hashAttribute = hashAttribute
        self.hashValue = hashValue
        self.featureId = featureId or None
        self.bucket = bucket
        self.stickyBucketUsed = stickyBucketUsed

        self.key = str(variationId)
        self.name = ""
        self.passthrough = False

        if meta:
            if "name" in meta:
                self.name = meta["name"]
            if "key" in meta:
                self.key = meta["key"]
            if "passthrough" in meta:
                self.passthrough = meta["passthrough"]

    def to_dict(self) -> dict:
        obj = {
            "featureId": self.featureId,
            "variationId": self.variationId,
            "inExperiment": self.inExperiment,
            "value": self.value,
            "hashUsed": self.hashUsed,
            "hashAttribute": self.hashAttribute,
            "hashValue": self.hashValue,
            "key": self.key,
            "stickyBucketUsed": self.stickyBucketUsed,
        }

        if self.bucket is not None:
            obj["bucket"] = self.bucket
        if self.name:
            obj["name"] = self.name
        if self.passthrough:
            obj["passthrough"] = True

        return obj


class Feature(object):
    def __init__(self, defaultValue=None, rules: list = []) -> None:
        self.defaultValue = defaultValue
        self.rules: List[FeatureRule] = []
        for rule in rules:
            if isinstance(rule, FeatureRule):
                self.rules.append(rule)
            else:
                self.rules.append(FeatureRule(
                    id=rule.get("id", None),
                    key=rule.get("key", ""),
                    variations=rule.get("variations", None),
                    weights=rule.get("weights", None),
                    coverage=rule.get("coverage", None),
                    condition=rule.get("condition", None),
                    namespace=rule.get("namespace", None),
                    force=rule.get("force", None),
                    hashAttribute=rule.get("hashAttribute", "id"),
                    fallbackAttribute=rule.get("fallbackAttribute", None),
                    hashVersion=rule.get("hashVersion", None),
                    range=rule.get("range", None),
                    ranges=rule.get("ranges", None),
                    meta=rule.get("meta", None),
                    filters=rule.get("filters", None),
                    seed=rule.get("seed", None),
                    name=rule.get("name", None),
                    phase=rule.get("phase", None),
                    disableStickyBucketing=rule.get("disableStickyBucketing", False),
                    bucketVersion=rule.get("bucketVersion", None),
                    minBucketVersion=rule.get("minBucketVersion", None),
                    parentConditions=rule.get("parentConditions", None),
                ))

    def to_dict(self) -> dict:
        return {
            "defaultValue": self.defaultValue,
            "rules": [rule.to_dict() for rule in self.rules],
        }


class FeatureRule(object):
    def __init__(
        self,
        id: str = None,
        key: str = "",
        variations: list = None,
        weights: List[float] = None,
        coverage: int = None,
        condition: dict = None,
        namespace: Tuple[str, float, float] = None,
        force=None,
        hashAttribute: str = "id",
        fallbackAttribute: str = None,
        hashVersion: int = None,
        range: Tuple[float, float] = None,
        ranges: List[Tuple[float, float]] = None,
        meta: List[VariationMeta] = None,
        filters: List[Filter] = None,
        seed: str = None,
        name: str = None,
        phase: str = None,
        disableStickyBucketing: bool = False,
        bucketVersion: int = None,
        minBucketVersion: int = None,
        parentConditions: List[dict] = None,
    ) -> None:

        if disableStickyBucketing:
            fallbackAttribute = None

        self.id = id
        self.key = key
        self.variations = variations
        self.weights = weights
        self.coverage = coverage
        self.condition = condition
        self.namespace = namespace
        self.force = force
        self.hashAttribute = hashAttribute
        self.fallbackAttribute = fallbackAttribute
        self.hashVersion = hashVersion or 1
        self.range = range
        self.ranges = ranges
        self.meta = meta
        self.filters = filters
        self.seed = seed
        self.name = name
        self.phase = phase
        self.disableStickyBucketing = disableStickyBucketing
        self.bucketVersion = bucketVersion or 0
        self.minBucketVersion = minBucketVersion or 0
        self.parentConditions = parentConditions

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {}
        if self.id:
            data["id"] = self.id
        if self.key:
            data["key"] = self.key
        if self.variations is not None:
            data["variations"] = self.variations
        if self.weights is not None:
            data["weights"] = self.weights
        if self.coverage and self.coverage != 1:
            data["coverage"] = self.coverage
        if self.condition is not None:
            data["condition"] = self.condition
        if self.namespace is not None:
            data["namespace"] = self.namespace
        if self.force is not None:
            data["force"] = self.force
        if self.hashAttribute != "id":
            data["hashAttribute"] = self.hashAttribute
        if self.hashVersion:
            data["hashVersion"] = self.hashVersion
        if self.range is not None:
            data["range"] = self.range
        if self.ranges is not None:
            data["ranges"] = self.ranges
        if self.meta is not None:
            data["meta"] = self.meta
        if self.filters is not None:
            data["filters"] = self.filters
        if self.seed is not None:
            data["seed"] = self.seed
        if self.name is not None:
            data["name"] = self.name
        if self.phase is not None:
            data["phase"] = self.phase
        if self.fallbackAttribute:
            data["fallbackAttribute"] = self.fallbackAttribute
        if self.disableStickyBucketing:
            data["disableStickyBucketing"] = True
        if self.bucketVersion:
            data["bucketVersion"] = self.bucketVersion
        if self.minBucketVersion:
            data["minBucketVersion"] = self.minBucketVersion
        if self.parentConditions:
            data["parentConditions"] = self.parentConditions

        return data


class FeatureResult(object):
    def __init__(
        self,
        value,
        source: str,
        experiment: Experiment = None,
        experimentResult: Result = None,
        ruleId: str = None,
    ) -> None:
        self.value = value
        self.source = source
        self.ruleId = ruleId
        self.experiment = experiment
        self.experimentResult = experimentResult
        self.on = bool(value)
        self.off = not bool(value)

    def to_dict(self) -> dict:
        data = {
            "value": self.value,
            "source": self.source,
            "on": self.on,
            "off": self.off,
        }
        if self.ruleId:
            data["ruleId"] = self.ruleId
        if self.experiment:
            data["experiment"] = self.experiment.to_dict()
        if self.experimentResult:
            data["experimentResult"] = self.experimentResult.to_dict()

        return data


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


class AbstractStickyBucketService(ABC):
    @abstractmethod
    def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def save_assignments(self, doc: Dict) -> None:
        pass

    def get_key(self, attributeName: str, attributeValue: str) -> str:
        return f"{attributeName}||{attributeValue}"

    # By default, just loop through all attributes and call get_assignments
    # Override this method in subclasses to perform a multi-query instead
    def get_all_assignments(self, attributes: Dict[str, str]) -> Dict[str, Dict]:
        docs = {}
        for attributeName, attributeValue in attributes.items():
            doc = self.get_assignments(attributeName, attributeValue)
            if doc:
                docs[self.get_key(attributeName, attributeValue)] = doc
        return docs


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

    def set_cache(self, cache: AbstractFeatureCache) -> None:
        self.cache = cache

    def clear_cache(self):
        self.cache.clear()

    def save_in_cache(self, key: str, res, ttl: int = 60):
        self.cache.set(key, res, ttl)

    # Loads features with an in-memory cache in front
    def load_features(
        self, api_host: str, client_key: str, decryption_key: str = "", ttl: int = 60
    ) -> Optional[Dict]:
        key = api_host + "::" + client_key

        cached = self.cache.get(key)
        if not cached:
            res = self._fetch_features(api_host, client_key, decryption_key)
            if res is not None:
                self.cache.set(key, res, ttl)
                logger.debug("Fetched features from API, stored in cache")
                return res
        return cached
    
    async def load_features_async(
        self, api_host: str, client_key: str, decryption_key: str = "", ttl: int = 60
    ) -> Optional[Dict]:
        key = api_host + "::" + client_key

        cached = self.cache.get(key)
        if not cached:
            res = await self._fetch_features_async(api_host, client_key, decryption_key)
            if res is not None:
                self.cache.set(key, res, ttl)
                logger.debug("Fetched features from API, stored in cache")
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
        cache_ttl: int = 60,
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

        if features:
            self.setFeatures(features)

        if self._streaming:
            self.load_features()
            self.startAutoRefresh()

    def load_features(self) -> None:
        if not self._client_key:
            raise ValueError("Must specify `client_key` to refresh features")

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

    def features_event_handler(self, features):
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
            
    def dispatch_sse_event(self, event_data):
        event_type = event_data['type']
        data = event_data['data']
        if event_type == 'features-updated':
            self.load_features()
        elif event_type == 'features':
            self.features_event_handler(data)


    def startAutoRefresh(self):
        if not self._client_key:
            raise ValueError("Must specify `client_key` to start features streaming")
       
        feature_repo.startAutoRefresh(
            api_host=self._api_host, 
            client_key=self._client_key,
            cb=self.dispatch_sse_event
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

    def eval_prereqs(self, parentConditions: List[dict], stack: Set[str]) -> str:
        for parentCondition in parentConditions:
            parentRes = self._eval_feature(parentCondition.get("id", None), stack)

            if parentRes.source == "cyclicPrerequisite":
                return "cyclic"

            if not evalCondition({'value': parentRes.value}, parentCondition.get("condition", None), self._saved_groups):
                if parentCondition.get("gate", False):
                    return "gate"
                return "fail"
        return "pass"

    def eval_feature(self, key: str) -> FeatureResult:
        return self._eval_feature(key, set())

    def _eval_feature(self, key: str, stack: Set[str]) -> FeatureResult:
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
                prereq_res = self.eval_prereqs(rule.parentConditions, stack)
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

            result = self._run(exp, key)
            self._fireSubscriptions(exp, result)

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

    # @deprecated, use get_all_results
    def getAllResults(self):
        return self.get_all_results()

    def get_all_results(self):
        return self._assigned.copy()

    def _getOrigHashValue(self, attr: str = None, fallbackAttr: str = None) -> Tuple[str, str]:
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

    def _getHashValue(self, attr: str = None, fallbackAttr: str = None) -> Tuple[str, str]:
        (attr, val) = self._getOrigHashValue(attr, fallbackAttr)
        return (attr, str(val))

    def _isIncludedInRollout(
        self,
        seed: str,
        hashAttribute: str = None,
        fallbackAttribute: str = None,
        range: Tuple[float, float] = None,
        coverage: float = None,
        hashVersion: int = None,
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

    def _fireSubscriptions(self, experiment: Experiment, result: Result):
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
        result = self._run(experiment)
        self._fireSubscriptions(experiment, result)
        return result

    def subscribe(self, callback):
        self._subscriptions.add(callback)
        return lambda: self._subscriptions.remove(callback)

    def _run(self, experiment: Experiment, featureId: Optional[str] = None) -> Result:
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
                prereq_res = self.eval_prereqs(experiment.parentConditions, set())
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
                self.sticky_bucket_service.save_assignments(doc)

        # 14. Fire the tracking callback if set
        self._track(experiment, result)

        # 15. Return the result
        logger.debug("Assigned variation %d in experiment %s", assigned, experiment.key)
        return result

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
        featureId: str = None,
        bucket: float = None,
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

    def _get_sticky_bucket_assignments(self, attr: str = None, fallback: str = None) -> Dict[str, str]:
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
        bucket_version: int = None,
        min_bucket_version: int = None,
        meta: List[VariationMeta] = None,
        hash_attribute: str = None,
        fallback_attribute: str = None
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

    def refresh_sticky_buckets(self, force: bool = False) -> None:
        if not self.sticky_bucket_service:
            return

        attributes = self._get_sticky_bucket_attributes()
        if not force and attributes == self._sticky_bucket_attributes:
            logger.debug("Skipping refresh of sticky bucket assignments, no changes")
            return

        self._sticky_bucket_attributes = attributes
        self._sticky_bucket_assignment_docs = self.sticky_bucket_service.get_all_assignments(attributes)

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
