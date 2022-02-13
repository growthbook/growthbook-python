#!/usr/bin/env python
"""
This is the Python client library for GrowthBook, the open-source
feature flagging and A/B testing platform.
More info at https://www.growthbook.io
"""

import re
from urllib.parse import urlparse, parse_qs
from typing import Optional


def fnv1a32(str: str) -> str:
    hval = 0x811C9DC5
    prime = 0x01000193
    uint32_max = 2 ** 32
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * prime) % uint32_max
    return hval


def gbhash(str: str) -> float:
    n = fnv1a32(str)
    return (n % 1000) / 1000


def inNamespace(userId: str, namespace: "tuple[str,float,float]") -> bool:
    n = gbhash(userId + "__" + namespace[0])
    return n >= namespace[1] and n < namespace[2]


def getEqualWeights(numVariations: int) -> "list[float]":
    if numVariations < 1:
        return []
    return [1 / numVariations for i in range(numVariations)]


def getBucketRanges(
    numVariations: int, coverage: float = 1, weights: "list[float]" = None
) -> "list[tuple[float,float]]":
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

    cumulative = 0
    ranges = []
    for w in weights:
        start = cumulative
        cumulative += w
        ranges.append((start, start + coverage * w))

    return ranges


def chooseVariation(n: float, ranges: "list[tuple[float,float]]") -> int:
    for i, r in enumerate(ranges):
        if n >= r[0] and n < r[1]:
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


class Experiment(object):
    def __init__(
        self,
        key: str,
        variations: list,
        weights: "list[float]" = None,
        active: bool = True,
        status: str = "running",
        coverage: int = 1,
        condition: dict = None,
        namespace: "tuple[str,float,float]" = None,
        url: str = "",
        include=None,
        groups: list = None,
        force: int = None,
        hashAttribute: str = "id",
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

        # Deprecated properties
        self.status = status
        self.url = url
        self.include = include
        self.groups = groups

    def to_dict(self):
        return {
            "key": self.key,
            "variations": self.variations,
            "weights": self.weights,
            "active": self.active,
            "coverage": self.coverage,
            "condition": self.condition,
            "namespace": self.namespace,
            "force": self.force,
            "hashAttribute": self.hashAttribute,
        }

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
        hashAttribute: str,
        hashValue: str,
    ) -> None:
        self.variationId = variationId
        self.inExperiment = inExperiment
        self.value = value
        self.hashAttribute = hashAttribute
        self.hashValue = hashValue

    def to_dict(self) -> dict:
        return {
            "variationId": self.variationId,
            "inExperiment": self.inExperiment,
            "value": self.value,
            "hashAttribute": self.hashAttribute,
            "hashValue": self.hashValue,
        }


class Feature(object):
    def __init__(self, defaultValue=None, rules: list = []) -> None:
        self.defaultValue = defaultValue
        self.rules: list[FeatureRule] = []
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
        weights: "list[float]" = None,
        coverage: int = 1,
        condition: dict = None,
        namespace: "tuple[str,float,float]" = None,
        force=None,
        hashAttribute: str = "id",
    ) -> None:
        self.key = key
        self.variations = variations
        self.weights = weights
        self.coverage = coverage
        self.condition = condition
        self.namespace = namespace
        self.force = force
        self.hashAttribute = hashAttribute

    def to_dict(self) -> dict:
        data = {}
        if self.key:
            data["key"] = self.key
        if self.variations is not None:
            data["variations"] = self.variations
        if self.weights is not None:
            data["weights"] = self.weights
        if self.coverage != 1:
            data["coverage"] = self.coverage
        if self.condition is not None:
            data["condition"] = self.condition
        if self.namespace is not None:
            data["namespace"] = self.namespace
        if self.force is not None:
            data["force"] = self.force
        if self.hashAttribute != "id":
            data["hashAttribute"] = self.hashAttribute

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


class GrowthBook(object):
    def __init__(
        self,
        enabled: bool = True,
        attributes: dict = {},
        url: str = "",
        features: dict = {},
        qaMode: bool = False,
        trackingCallback=None,
        # Deprecated args
        user: dict = {},
        groups: dict = {},
        overrides: dict = {},
        forcedVariations: dict = {},
    ):
        self._enabled = enabled
        self._attributes = attributes
        self._url = url
        self._features: dict[str, Feature] = {}

        if features:
            self.setFeatures(features)

        self._qaMode = qaMode
        self._trackingCallback = trackingCallback

        # Deprecated args
        self._user = user
        self._groups = groups
        self._overrides = overrides
        self._forcedVariations = forcedVariations

        self._tracked = {}
        self._assigned = {}
        self._subscriptions = set()

    def setFeatures(self, features: dict) -> None:
        self._features = {}
        for key, feature in features.items():
            if isinstance(feature, Feature):
                self._features[key] = feature
            else:
                self._features[key] = Feature(**feature)

    def getFeatures(self) -> "dict[str,Feature]":
        return self._features

    def setAttributes(self, attributes: dict) -> None:
        self._attributes = attributes

    def getAttributes(self) -> dict:
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

    def isOn(self, key: str) -> bool:
        return self.evalFeature(key).on

    def isOff(self, key: str) -> bool:
        return self.evalFeature(key).off

    def getFeatureValue(self, key: str, fallback):
        res = self.evalFeature(key)
        return res.value if res.value is not None else fallback

    def evalFeature(self, key: str) -> FeatureResult:
        if key not in self._features:
            return FeatureResult(None, "unknownFeature")

        feature = self._features[key]
        for rule in feature.rules:
            if rule.condition:
                if not evalCondition(self._attributes, rule.condition):
                    continue
            if rule.force is not None:
                if rule.coverage < 1:
                    hashValue = self._getHashValue(rule.hashAttribute)
                    if not hashValue:
                        continue

                    n = gbhash(hashValue + key)

                    if n > rule.coverage:
                        continue
                return FeatureResult(rule.force, "force")

            if rule.variations is None:
                continue

            exp = Experiment(
                key=rule.key or key,
                variations=rule.variations,
                coverage=rule.coverage,
                weights=rule.weights,
                hashAttribute=rule.hashAttribute,
                namespace=rule.namespace,
            )

            result = self.run(exp)

            if not result.inExperiment:
                continue

            return FeatureResult(result.value, "experiment", exp, result)

        return FeatureResult(feature.defaultValue, "defaultValue")

    def getAllResults(self):
        return self._assigned.copy()

    def _getHashValue(self, attr: str) -> str:
        if attr in self._attributes:
            return str(self._attributes[attr])
        if attr in self._user:
            return str(self._user[attr])
        return ""

    def run(self, experiment: Experiment) -> Result:
        result = self._run(experiment)

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

        return result

    def subscribe(self, callback):
        self._subscriptions.add(callback)
        return lambda: self._subscriptions.remove(callback)

    def _run(self, experiment: Experiment) -> Result:
        # 1. If experiment has less than 2 variations, return immediately
        if len(experiment.variations) < 2:
            return self._getExperimentResult(experiment)
        # 2. If growthbook is disabled, return immediately
        if not self._enabled:
            return self._getExperimentResult(experiment)
        # 2.5. If the experiment props have been overridden, merge them in
        if self._overrides.get(experiment.key, None):
            experiment.update(self._overrides[experiment.key])
        # 3. If experiment is forced via a querystring in the url
        qs = getQueryStringOverride(
            experiment.key, self._url, len(experiment.variations)
        )
        if qs is not None:
            return self._getExperimentResult(experiment, qs)
        # 4. If variation is forced in the context
        if self._forcedVariations.get(experiment.key, None) is not None:
            return self._getExperimentResult(
                experiment, self._forcedVariations[experiment.key]
            )
        # 5. If experiment is a draft or not active, return immediately
        if experiment.status == "draft" or not experiment.active:
            return self._getExperimentResult(experiment)
        # 6. Get the user hash attribute and value
        hashAttribute = experiment.hashAttribute or "id"
        hashValue = self._getHashValue(hashAttribute)
        if not hashValue:
            return self._getExperimentResult(experiment)

        # 7. Exclude if user not in experiment.namespace
        if experiment.namespace and not inNamespace(hashValue, experiment.namespace):
            return self._getExperimentResult(experiment)

        # 7.5. If experiment has an include property
        if experiment.include:
            try:
                if not experiment.include():
                    return self._getExperimentResult(experiment)
            except Exception:
                return self._getExperimentResult(experiment)

        # 8. Exclude if condition is false
        if experiment.condition and not evalCondition(
            self._attributes, experiment.condition
        ):
            return self._getExperimentResult(experiment)

        # 8.1. Make sure user is in a matching group
        if experiment.groups and len(experiment.groups):
            expGroups = self._groups or {}
            matched = False
            for group in experiment.groups:
                if expGroups[group]:
                    matched = True
            if not matched:
                return self._getExperimentResult(experiment)
        # 8.2. If experiment.url is set, see if it's valid
        if experiment.url:
            if not self._urlIsValid(experiment.url):
                return self._getExperimentResult(experiment)

        # 9. Get bucket ranges and choose variation
        ranges = getBucketRanges(
            len(experiment.variations), experiment.coverage or 1, experiment.weights
        )
        n = gbhash(hashValue + experiment.key)
        assigned = chooseVariation(n, ranges)

        # 10. Return if not in experiment
        if assigned < 0:
            return self._getExperimentResult(experiment)

        # 11. If experiment is forced, return immediately
        if experiment.force is not None:
            return self._getExperimentResult(experiment, experiment.force)

        # 12. Exclude if in QA mode
        if self._qaMode:
            return self._getExperimentResult(experiment)

        # 12.5. If experiment is stopped, return immediately
        if experiment.status == "stopped":
            return self._getExperimentResult(experiment)

        # 13. Build the result object
        result = self._getExperimentResult(experiment, assigned, True)

        # 14. Fire the tracking callback if set
        self._track(experiment, result)

        # 15. Return the result
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
        self, experiment: Experiment, variationId: int = 0, inExperiment: bool = False
    ) -> Result:
        hashAttribute = experiment.hashAttribute or "id"

        if variationId < 0 or variationId > len(experiment.variations) - 1:
            variationId = 0

        return Result(
            inExperiment=inExperiment,
            variationId=variationId,
            value=experiment.variations[variationId],
            hashAttribute=hashAttribute,
            hashValue=self._getHashValue(hashAttribute),
        )
