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
    return n >= range[0] and n < range[1]


def inNamespace(userId: str, namespace: Tuple[str, float, float]) -> bool:
    n = gbhash("__" + namespace[0], userId, 1)
    if n is None:
        return False
    return n >= namespace[1] and n < namespace[2]


def getEqualWeights(numVariations: int) -> List[float]:
    if numVariations < 1:
        return []
    return [1 / numVariations for i in range(numVariations)]


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


def evalCondition(attributes: dict, condition: dict) -> bool:
    if "$or" in condition:
        return evalOr(attributes, condition["$or"])
    if "$nor" in condition:
        return not evalOr(attributes, condition["$nor"])
    if "$and" in condition:
        return evalAnd(attributes, condition["$and"])
    if "$not" in condition:
        return not evalCondition(attributes, condition["$not"])

    for key, value in condition.items():
        if not evalConditionValue(value, getPath(attributes, key)):
            return False

    return True


def evalOr(attributes, conditions) -> bool:
    if len(conditions) == 0:
        return True

    for condition in conditions:
        if evalCondition(attributes, condition):
            return True
    return False


def evalAnd(attributes, conditions) -> bool:
    for condition in conditions:
        if not evalCondition(attributes, condition):
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


def evalConditionValue(conditionValue, attributeValue) -> bool:
    if type(conditionValue) is dict and isOperatorObject(conditionValue):
        for key, value in conditionValue.items():
            if not evalOperatorCondition(key, attributeValue, value):
                return False
        return True
    return conditionValue == attributeValue


def elemMatch(condition, attributeValue) -> bool:
    if not type(attributeValue) is list:
        return False

    for item in attributeValue:
        if isOperatorObject(condition):
            if evalConditionValue(condition, item):
                return True
        else:
            if evalCondition(item, condition):
                return True

    return False


def evalOperatorCondition(operator, attributeValue, conditionValue) -> bool:
    if operator == "$eq":
        return attributeValue == conditionValue
    elif operator == "$ne":
        return attributeValue != conditionValue
    elif operator == "$lt":
        return attributeValue < conditionValue
    elif operator == "$lte":
        return attributeValue <= conditionValue
    elif operator == "$gt":
        return attributeValue > conditionValue
    elif operator == "$gte":
        return attributeValue >= conditionValue
    elif operator == "$regex":
        try:
            r = re.compile(conditionValue)
            return bool(r.search(attributeValue))
        except Exception:
            return False
    elif operator == "$in":
        return attributeValue in conditionValue
    elif operator == "$nin":
        return not (attributeValue in conditionValue)
    elif operator == "$elemMatch":
        return elemMatch(conditionValue, attributeValue)
    elif operator == "$size":
        if not (type(attributeValue) is list):
            return False
        return evalConditionValue(conditionValue, len(attributeValue))
    elif operator == "$all":
        if not (type(attributeValue) is list):
            return False
        for cond in conditionValue:
            passing = False
            for attr in attributeValue:
                if evalConditionValue(cond, attr):
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
        return not evalConditionValue(conditionValue, attributeValue)
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
        hashVersion: int = None,
        ranges: List[Tuple[float, float]] = None,
        meta: List[VariationMeta] = None,
        filters: List[Filter] = None,
        seed: str = None,
        name: str = None,
        phase: str = None,
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
    ) -> None:
        self.variationId = variationId
        self.inExperiment = inExperiment
        self.value = value
        self.hashUsed = hashUsed
        self.hashAttribute = hashAttribute
        self.hashValue = hashValue
        self.featureId = featureId or None
        self.bucket = bucket

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
                self.rules.append(FeatureRule(**rule))

    def to_dict(self) -> dict:
        return {
            "defaultValue": self.defaultValue,
            "rules": [rule.to_dict() for rule in self.rules],
        }


class FeatureRule(object):
    def __init__(
        self,
        key: str = "",
        variations: list = None,
        weights: List[float] = None,
        coverage: int = None,
        condition: dict = None,
        namespace: Tuple[str, float, float] = None,
        force=None,
        hashAttribute: str = "id",
        hashVersion: int = None,
        range: Tuple[float, float] = None,
        ranges: List[Tuple[float, float]] = None,
        meta: List[VariationMeta] = None,
        filters: List[Filter] = None,
        seed: str = None,
        name: str = None,
        phase: str = None,
    ) -> None:
        self.key = key
        self.variations = variations
        self.weights = weights
        self.coverage = coverage
        self.condition = condition
        self.namespace = namespace
        self.force = force
        self.hashAttribute = hashAttribute
        self.hashVersion = hashVersion or 1
        self.range = range
        self.ranges = ranges
        self.meta = meta
        self.filters = filters
        self.seed = seed
        self.name = name
        self.phase = phase

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {}
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

        return data


class FeatureResult(object):
    def __init__(
        self,
        value,
        source: str,
        experiment: Experiment = None,
        experimentResult: Result = None,
    ) -> None:
        self.value = value
        self.source = source
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


class FeatureRepository(object):
    def __init__(self) -> None:
        self.cache: AbstractFeatureCache = InMemoryFeatureCache()
        self.http: Optional[PoolManager] = None

    def set_cache(self, cache: AbstractFeatureCache) -> None:
        self.cache = cache

    def clear_cache(self):
        self.cache.clear()

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

    # Fetch features from the GrowthBook API
    def _fetch_features(
        self, api_host: str, client_key: str, decryption_key: str = ""
    ) -> Optional[Dict]:
        decoded = self._fetch_and_decode(api_host, client_key)
        if not decoded:
            return None

        if "encryptedFeatures" in decoded:
            if not decryption_key:
                raise ValueError("Must specify decryption_key")
            try:
                decrypted = decrypt(decoded["encryptedFeatures"], decryption_key)
                return json.loads(decrypted)
            except Exception:
                logger.warning(
                    "Failed to decrypt features from GrowthBook API response"
                )
                return None
        elif "features" in decoded:
            return decoded["features"]
        else:
            logger.warning("GrowthBook API response missing features")
            return None

    def _get_features_url(self, api_host: str, client_key: str) -> str:
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
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl

        if features:
            self.setFeatures(features)

        self._qaMode = qa_mode or qaMode
        self._trackingCallback = on_experiment_viewed or trackingCallback

        # Deprecated args
        self._user = user
        self._groups = groups
        self._overrides = overrides
        self._forcedVariations = forced_variations or forcedVariations

        self._tracked: Dict[str, Any] = {}
        self._assigned: Dict[str, Any] = {}
        self._subscriptions: Set[Any] = set()

    def load_features(self) -> None:
        if not self._client_key:
            raise ValueError("Must specify `client_key` to refresh features")

        features = feature_repo.load_features(
            self._api_host, self._client_key, self._decryption_key, self._cache_ttl
        )
        if features is not None:
            self.setFeatures(features)

    # @deprecated, use set_features
    def setFeatures(self, features: dict) -> None:
        return self.set_features(features)

    def set_features(self, features: dict) -> None:
        self._features = {}
        for key, feature in features.items():
            if isinstance(feature, Feature):
                self._features[key] = feature
            else:
                self._features[key] = Feature(**feature)

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

    def eval_feature(self, key: str) -> FeatureResult:
        logger.debug("Evaluating feature %s", key)
        if key not in self._features:
            logger.warning("Unknown feature %s", key)
            return FeatureResult(None, "unknownFeature")

        feature = self._features[key]
        for rule in feature.rules:
            if rule.condition:
                if not evalCondition(self._attributes, rule.condition):
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
                return FeatureResult(rule.force, "force")

            if rule.variations is None:
                logger.warning("Skip invalid rule, feature %s", key)
                continue

            exp = Experiment(
                key=rule.key or key,
                variations=rule.variations,
                coverage=rule.coverage,
                weights=rule.weights,
                hashAttribute=rule.hashAttribute,
                namespace=rule.namespace,
                hashVersion=rule.hashVersion,
                meta=rule.meta,
                ranges=rule.ranges,
                name=rule.name,
                phase=rule.phase,
                seed=rule.seed,
                filters=rule.filters,
            )

            result = self._run(exp, key)
            self._fireSubscriptions(exp, result)

            if not result.inExperiment:
                logger.debug("Skip rule because user not included in experiment", key)
                continue

            if result.passthrough:
                logger.debug("Continue to next rule, feature %s", key)
                continue

            logger.debug("Assign value from experiment, feature %s", key)
            return FeatureResult(result.value, "experiment", exp, result)

        logger.debug("Use default value for feature %s", key)
        return FeatureResult(feature.defaultValue, "defaultValue")

    # @deprecated, use get_all_results
    def getAllResults(self):
        return self.get_all_results()

    def get_all_results(self):
        return self._assigned.copy()

    def _getOrigHashValue(self, attr: str = None):
        attr = attr or "id"
        if attr in self._attributes:
            return self._attributes[attr] or ""
        if attr in self._user:
            return self._user[attr] or ""
        return ""

    def _getHashValue(self, attr: str = None) -> str:
        return str(self._getOrigHashValue(attr))

    def _isIncludedInRollout(
        self,
        seed: str,
        hashAttribute: str = None,
        range: Tuple[float, float] = None,
        coverage: float = None,
        hashVersion: int = None,
    ) -> bool:
        if coverage is None and range is None:
            return True

        hash_value = self._getHashValue(hashAttribute or "id")
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
            hash_value = self._getHashValue(filter.get("attribute", "id"))
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
        hashAttribute = experiment.hashAttribute or "id"
        hashValue = self._getHashValue(hashAttribute)
        if not hashValue:
            logger.debug(
                "Skip experiment %s because user's hashAttribute value is empty",
                experiment.key,
            )
            return self._getExperimentResult(experiment, featureId=featureId)

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
            self._attributes, experiment.condition
        ):
            logger.debug(
                "Skip experiment %s because user failed the condition", experiment.key
            )
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
        # 8.2. If experiment.url is set, see if it's valid
        if experiment.url:
            if not self._urlIsValid(experiment.url):
                logger.debug(
                    "Skip experiment %s because current URL is not targeted",
                    experiment.key,
                )
                return self._getExperimentResult(experiment, featureId=featureId)

        # 9. Get bucket ranges and choose variation
        c = experiment.coverage
        ranges = experiment.ranges or getBucketRanges(
            len(experiment.variations), c if c is not None else 1, experiment.weights
        )
        n = gbhash(
            experiment.seed or experiment.key, hashValue, experiment.hashVersion or 1
        )
        if n is None:
            logger.warning(
                "Skip experiment %s because of invalid hashVersion", experiment.key
            )
            return self._getExperimentResult(experiment, featureId=featureId)
        assigned = chooseVariation(n, ranges)

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
            experiment, assigned, True, featureId=featureId, bucket=n
        )

        # 14. Fire the tracking callback if set
        self._track(experiment, result)

        # 15. Return the result
        logger.debug(
            "Assigned variation %d in experiment %s", assigned, experiment.key
        )
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
    ) -> Result:
        hashAttribute = experiment.hashAttribute or "id"

        inExperiment = True
        if variationId < 0 or variationId > len(experiment.variations) - 1:
            variationId = 0
            inExperiment = False

        meta = None
        if experiment.meta:
            meta = experiment.meta[variationId]

        return Result(
            featureId=featureId,
            inExperiment=inExperiment,
            variationId=variationId,
            value=experiment.variations[variationId],
            hashUsed=hashUsed,
            hashAttribute=hashAttribute,
            hashValue=self._getOrigHashValue(hashAttribute),
            meta=meta,
            bucket=bucket,
        )
