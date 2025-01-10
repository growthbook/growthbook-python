import logging
import re
import json

from urllib.parse import urlparse, parse_qs
from typing import Callable, Optional, Any, Set, Tuple, List, Dict
from .common_types import EvaluationContext, FeatureResult, Experiment, Filter, Result, VariationMeta


logger = logging.getLogger("growthbook.core")

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

def _getOrigHashValue(attr: Optional[str] = None, 
                      fallbackAttr: Optional[str] = None, 
                      eval_context: EvaluationContext = None
                    ) -> Tuple[str, str]:
    # attr = attr or "id" -- Fix for the flaky behavior of sticky bucket assignment
    val = ""

    if attr in eval_context.user.attributes:
        val = "" if eval_context.user.attributes[attr] is None else eval_context.user.attributes[attr]

    # If no match, try fallback
    if (not val or val == "") and fallbackAttr and eval_context.global_ctx.options.sticky_bucket_service:
        if fallbackAttr in eval_context.user.attributes:
            val = "" if eval_context.user.attributes[fallbackAttr] is None else eval_context.user.attributes[fallbackAttr]

        if not val or val != "":
            attr = fallbackAttr

    return (attr, val)

def _getHashValue(attr: str = None, fallbackAttr: str = None, eval_context: EvaluationContext = None) -> Tuple[str, str]:
    (attr, val) = _getOrigHashValue(attr=attr, fallbackAttr=fallbackAttr, eval_context=eval_context)
    return (attr, str(val))

def _isIncludedInRollout(
    seed: str,
    hashAttribute: str = None,
    fallbackAttribute: str = None,
    range: Tuple[float, float] = None,
    coverage: float = None,
    hashVersion: int = None,
    eval_context: EvaluationContext = None
) -> bool:
    if coverage is None and range is None:
        return True

    (_, hash_value) = _getHashValue(attr=hashAttribute, fallbackAttr=fallbackAttribute, eval_context=eval_context)
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

def _isFilteredOut(filters: List[Filter], eval_context: EvaluationContext) -> bool:
    for filter in filters:
        (_, hash_value) = _getHashValue(attr=filter.get("attribute", "id"), eval_context=eval_context)  
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


def fnv1a32(str: str) -> int:
    hval = 0x811C9DC5
    prime = 0x01000193
    uint32_max = 2 ** 32
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * prime) % uint32_max
    return hval

def inNamespace(userId: str, namespace: Tuple[str, float, float]) -> bool:
    n = gbhash("__" + namespace[0], userId, 1)
    if n is None:
        return False
    return namespace[1] <= n < namespace[2]

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

def _urlIsValid(url: str, pattern: str) -> bool:
    if not url: # it was self._url! Ignored the param passed in.
        return False

    try:
        r = re.compile(pattern)
        if r.search(url):
            return True

        pathOnly = re.sub(r"^[^/]*/", "/", re.sub(r"^https?:\/\/", "", url))
        if r.search(pathOnly):
            return True
        return False
    except Exception:
        return True

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

def eval_feature(
    key: str,
    evalContext: EvaluationContext = None,
    callback_subscription: Callable[[Experiment, Result], None] = None
) -> FeatureResult:
    """Core feature evaluation logic as a standalone function"""

    if evalContext is None:
        raise ValueError("evalContext is required - eval_feature")
    
    if key not in evalContext.global_ctx.features:
        logger.warning("Unknown feature %s", key)
        return FeatureResult(None, "unknownFeature")

    if key in evalContext.stack.evaluted_features:
        logger.warning("Cyclic prerequisite detected, stack: %s", evalContext.stack.evaluted_features)
        return FeatureResult(None, "cyclicPrerequisite")
 
    evalContext.stack.evaluted_features.add(key)

    feature = evalContext.global_ctx.features[key]

    for rule in feature.rules:
        if (rule.parentConditions):
            prereq_res = eval_prereqs(parentConditions=rule.parentConditions, evalContext=evalContext)
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
            if not evalCondition(evalContext.user.attributes, rule.condition, evalContext.global_ctx.saved_groups):
                logger.debug(
                    "Skip rule because of failed condition, feature %s", key
                )
                continue
        if rule.filters:
            if _isFilteredOut(rule.filters, evalContext):
                logger.debug(
                    "Skip rule because of filters/namespaces, feature %s", key
                )
                continue
        if rule.force is not None:
            if not _isIncludedInRollout(
                seed=rule.seed or key,
                hashAttribute=rule.hashAttribute,
                fallbackAttribute=rule.fallbackAttribute,
                range=rule.range,
                coverage=rule.coverage,
                hashVersion=rule.hashVersion,
                eval_context=evalContext
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

        result = run_experiment(experiment=exp, featureId=key, evalContext=evalContext)

        if callback_subscription:
            callback_subscription(exp, result)

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

def eval_prereqs(parentConditions: List[dict], evalContext: EvaluationContext) -> str:
    for parentCondition in parentConditions:
        parentRes = eval_feature(key=parentCondition.get("id", None), evalContext=evalContext)

        if parentRes.source == "cyclicPrerequisite":
            return "cyclic"

        if not evalCondition({'value': parentRes.value}, parentCondition.get("condition", None), evalContext.global_ctx.saved_groups):
            if parentCondition.get("gate", False):
                return "gate"
            return "fail"
    return "pass"

def _get_sticky_bucket_experiment_key(experiment_key: str, bucket_version: int = 0) -> str:
    return experiment_key + "__" + str(bucket_version)
    
def _get_sticky_bucket_assignments(attr: str = None, fallback: str = None,
                                    evalContext: EvaluationContext = None) -> Dict[str, str]:
    merged: Dict[str, str] = {}

    # Search for docs stored for attribute(id)
    _, hashValue = _getHashValue(attr=attr, eval_context=evalContext)
    key = f"{attr}||{hashValue}"
    if key in evalContext.user.sticky_bucket_assignment_docs:
        merged = evalContext.user.sticky_bucket_assignment_docs[key].get("assignments", {})

    # Search for docs stored for fallback attribute
    if fallback:
        _, hashValue = _getHashValue(fallbackAttr=fallback, eval_context=evalContext)
        key = f"{fallback}||{hashValue}"
        if key in evalContext.user.sticky_bucket_assignment_docs:
            # Merge the fallback assignments, but don't overwrite existing ones
            for k, v in evalContext.user.sticky_bucket_assignment_docs[key].get("assignments", {}).items():
                if k not in merged:
                    merged[k] = v

    return merged

def _is_blocked(
    assignments: Dict[str, str],
    experiment_key: str,
    min_bucket_version: int
) -> bool:
    if min_bucket_version > 0:
        for i in range(min_bucket_version):
            blocked_key = _get_sticky_bucket_experiment_key(experiment_key, i)
            if blocked_key in assignments:
                return True
    return False

def _get_sticky_bucket_variation(
    experiment_key: str,
    bucket_version: int = None,
    min_bucket_version: int = None,
    meta: List[VariationMeta] = None,
    hash_attribute: str = None,
    fallback_attribute: str = None,
    evalContext: EvaluationContext = None
) -> dict:
    bucket_version = bucket_version or 0
    min_bucket_version = min_bucket_version or 0
    meta = meta or []

    id = _get_sticky_bucket_experiment_key(experiment_key, bucket_version)

    assignments = _get_sticky_bucket_assignments(attr=hash_attribute, fallback=fallback_attribute, evalContext=evalContext)
    if _is_blocked(assignments, experiment_key, min_bucket_version):
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

def run_experiment(experiment: Experiment, 
                   featureId: Optional[str] = None, 
                   evalContext: EvaluationContext = None, 
                   tracking_cb: Callable[[Experiment, Result], None] = None
                ) -> Result:
    if evalContext is None:
        raise ValueError("evalContext is required - run_experiment")
    # 1. If experiment has less than 2 variations, return immediately
    if len(experiment.variations) < 2:
        logger.warning(
            "Experiment %s has less than 2 variations, skip", experiment.key
        )
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)
    # 2. If growthbook is disabled, return immediately
    if not evalContext.global_ctx.options.enabled:
        logger.debug(
            "Skip experiment %s because GrowthBook is disabled", experiment.key
        )
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)
    # 2.5. If the experiment props have been overridden, merge them in
    if evalContext.user.overrides.get(experiment.key, None):
        experiment.update(evalContext.user.overrides[experiment.key])
    # 3. If experiment is forced via a querystring in the url
    qs = getQueryStringOverride(
        experiment.key, evalContext.user.url, len(experiment.variations)
    )
    if qs is not None:
        logger.debug(
            "Force variation %d from URL querystring, experiment %s",
            qs,
            experiment.key,
        )
        return _getExperimentResult(experiment=experiment, variationId=qs, featureId=featureId, evalContext=evalContext)
    # 4. If variation is forced in the context
    if evalContext.user.forced_variations.get(experiment.key, None) is not None:
        logger.debug(
            "Force variation %d from GrowthBook context, experiment %s",
            evalContext.user.forced_variations[experiment.key],
            experiment.key,
        )
        return _getExperimentResult(
            experiment=experiment, variationId=evalContext.user.forced_variations[experiment.key], featureId=featureId, evalContext=evalContext
        )
    # 5. If experiment is a draft or not active, return immediately
    if experiment.status == "draft" or not experiment.active:
        logger.debug("Experiment %s is not active, skip", experiment.key)
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    # 6. Get the user hash attribute and value
    (hashAttribute, hashValue) = _getHashValue(attr=experiment.hashAttribute, fallbackAttr=experiment.fallbackAttribute, eval_context=evalContext)
    if not hashValue:
        logger.debug(
            "Skip experiment %s because user's hashAttribute value is empty",
            experiment.key,
        )
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    assigned = -1

    found_sticky_bucket = False
    sticky_bucket_version_is_blocked = False
    if evalContext.global_ctx.options.sticky_bucket_service and not experiment.disableStickyBucketing:
        sticky_bucket = _get_sticky_bucket_variation(
            experiment.key,
            experiment.bucketVersion,
            experiment.minBucketVersion,
            experiment.meta,
            hash_attribute=experiment.hashAttribute,
            fallback_attribute=experiment.fallbackAttribute,
            evalContext=evalContext
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
            if _isFilteredOut(experiment.filters, evalContext):
                logger.debug(
                    "Skip experiment %s because of filters/namespaces", experiment.key
                )
                return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)
        elif experiment.namespace and not inNamespace(hashValue, experiment.namespace):
            logger.debug("Skip experiment %s because of namespace", experiment.key)
            return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

        # 7.5. If experiment has an include property
        if experiment.include:
            try:
                if not experiment.include():
                    logger.debug(
                        "Skip experiment %s because include() returned false",
                        experiment.key,
                    )
                    return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)
            except Exception:
                logger.warning(
                    "Skip experiment %s because include() raised an Exception",
                    experiment.key,
                )
                return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

        # 8. Exclude if condition is false
        if experiment.condition and not evalCondition(
            evalContext.user.attributes, experiment.condition, evalContext.global_ctx.saved_groups
        ):
            logger.debug(
                "Skip experiment %s because user failed the condition", experiment.key
            )
            return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

        # 8.05 Exclude if parent conditions are not met
        if (experiment.parentConditions):
            prereq_res = eval_prereqs(parentConditions=experiment.parentConditions, evalContext=evalContext)
            if prereq_res == "gate" or prereq_res == "fail":
                logger.debug("Skip experiment %s because of failing prerequisite", experiment.key)
                return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)
            if prereq_res == "cyclic":
                logger.debug("Skip experiment %s because of cyclic prerequisite", experiment.key)
                return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

        # 8.1. Make sure user is in a matching group
        if experiment.groups and len(experiment.groups):
            expGroups = evalContext.user.groups or {}
            matched = False
            for group in experiment.groups:
                if expGroups[group]:
                    matched = True
            if not matched:
                logger.debug(
                    "Skip experiment %s because user not in required group",
                    experiment.key,
                )
                return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    # The following apply even when in a sticky bucket

    # 8.2. If experiment.url is set, see if it's valid
    if experiment.url:
        if not _urlIsValid(url=evalContext.global_ctx.options.url, pattern=experiment.url):
            logger.debug(
                "Skip experiment %s because current URL is not targeted",
                experiment.key,
            )
            return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    # 9. Get bucket ranges and choose variation
    n = gbhash(
        experiment.seed or experiment.key, hashValue, experiment.hashVersion or 1
    )
    if n is None:
        logger.warning(
            "Skip experiment %s because of invalid hashVersion", experiment.key
        )
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    if not found_sticky_bucket:
        c = experiment.coverage
        ranges = experiment.ranges or getBucketRanges(
            len(experiment.variations), c if c is not None else 1, experiment.weights
        )
        assigned = chooseVariation(n, ranges)

    # Unenroll if any prior sticky buckets are blocked by version
    if sticky_bucket_version_is_blocked:
        logger.debug("Skip experiment %s because sticky bucket version is blocked", experiment.key)
        return _getExperimentResult(experiment=experiment, featureId=featureId, stickyBucketUsed=True, evalContext=evalContext)

    # 10. Return if not in experiment
    if assigned < 0:
        logger.debug(
            "Skip experiment %s because user is not included in the rollout",
            experiment.key,
        )
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    # 11. If experiment is forced, return immediately
    if experiment.force is not None:
        logger.debug(
            "Force variation %d in experiment %s", experiment.force, experiment.key
        )
        return _getExperimentResult(
            experiment=experiment, variationId=experiment.force, featureId=featureId, evalContext=evalContext
        )

    # 12. Exclude if in QA mode
    if evalContext.global_ctx.options.qa_mode:
        logger.debug("Skip experiment %s because of QA Mode", experiment.key)
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    # 12.5. If experiment is stopped, return immediately
    if experiment.status == "stopped":
        logger.debug("Skip experiment %s because it is stopped", experiment.key)
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    # 13. Build the result object
    result = _getExperimentResult(
        experiment=experiment, variationId=assigned, hashUsed=True, featureId=featureId, bucket=n, stickyBucketUsed=found_sticky_bucket, evalContext=evalContext
    )

    # 13.5 Persist sticky bucket
    if evalContext.global_ctx.options.sticky_bucket_service and not experiment.disableStickyBucketing:
        assignment = {}
        assignment[_get_sticky_bucket_experiment_key(
            experiment.key,
            experiment.bucketVersion
        )] = result.key

        data = _generate_sticky_bucket_assignment_doc(
            attribute_name=hashAttribute,
            attribute_value=hashValue,
            assignments=assignment,
            evalContext=evalContext
        )
        doc = data.get("doc", None)
        if doc and data.get('changed', False):
            if not evalContext.user.sticky_bucket_assignment_docs:
                evalContext.user.sticky_bucket_assignment_docs = {}
            evalContext.user.sticky_bucket_assignment_docs[data.get('key')] = doc
            evalContext.global_ctx.options.sticky_bucket_service.save_assignments(doc)

    # 14. Fire the tracking callback if set
    if tracking_cb:
        tracking_cb(experiment, result)

    # 15. Return the result
    logger.debug("Assigned variation %d in experiment %s", assigned, experiment.key)
    return result

def _generate_sticky_bucket_assignment_doc(attribute_name: str, attribute_value: str, assignments: dict, evalContext: EvaluationContext):
    key = attribute_name + "||" + attribute_value
    existing_assignments = evalContext.user.sticky_bucket_assignment_docs.get(key, {}).get("assignments", {})

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
    
def _getExperimentResult(
    experiment: Experiment,
    variationId: int = -1,
    hashUsed: bool = False,
    featureId: str = None,
    bucket: float = None,
    stickyBucketUsed: bool = False,
    evalContext: EvaluationContext = None
) -> Result:
    inExperiment = True
    if variationId < 0 or variationId > len(experiment.variations) - 1:
        variationId = 0
        inExperiment = False

    meta = None
    if experiment.meta:
        meta = experiment.meta[variationId]

    (hashAttribute, hashValue) = _getOrigHashValue(attr=experiment.hashAttribute, fallbackAttr=experiment.fallbackAttribute, eval_context=evalContext)

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