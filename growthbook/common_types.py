#!/usr/bin/env python

import sys
# Only require typing_extensions if using Python 3.7 or earlier
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from enum import Enum
from abc import ABC, abstractmethod

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

@dataclass
class StackContext: 
    id: Optional[str] = None
    evaluted_features: Set[str] = field(default_factory=set)

class FeatureRefreshStrategy(Enum):
    STALE_WHILE_REVALIDATE = 'HTTP_REFRESH'
    SERVER_SENT_EVENTS = 'SSE'

@dataclass
class Options:
    url: Optional[str] = None
    api_host: Optional[str] = "https://cdn.growthbook.io"
    client_key: Optional[str] = None
    decryption_key: Optional[str] = None
    cache_ttl: int = 60
    enabled: bool = True
    qa_mode: bool = False
    enable_dev_mode: bool = False
    # forced_variations: Dict[str, Any] = field(default_factory=dict)
    refresh_strategy: Optional[FeatureRefreshStrategy] = FeatureRefreshStrategy.STALE_WHILE_REVALIDATE
    sticky_bucket_service: AbstractStickyBucketService = None
    sticky_bucket_identifier_attributes: List[str] = None
    on_experiment_viewed=None

@dataclass
class UserContext:
    # user_id: Optional[str] = None
    url: Optional[str] = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    groups: Dict[str, str] = field(default_factory=dict)
    forced_variations: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    sticky_bucket_assignment_docs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GlobalContext:
    options: Options
    features: Dict[str, Any] = field(default_factory=dict)
    saved_groups: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationContext:
    user: UserContext
    global_ctx: GlobalContext
    stack: StackContext
