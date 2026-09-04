import inspect
import logging
import math
import re
import json
from functools import lru_cache

from urllib.parse import urlparse, parse_qs
from typing import Callable, Optional, Any, Set, Tuple, List, Dict, cast
from typing_extensions import TypeGuard
from .common_types import (
    ContextualBanditAssignment,
    ContextualBanditContext,
    EvaluationContext,
    FeatureResult,
    Experiment,
    Filter,
    Result,
    UserContext,
    VariationMeta,
)


logger = logging.getLogger("growthbook.core")

# leafId reported when a contextual bandit rule falls back to its marginal
# weights (no leaf condition matched, empty contexts, or leaf selection
# errored). Matches CONTEXTUAL_BANDIT_FALLBACK_LEAF_ID in the JS SDK.
CONTEXTUAL_BANDIT_FALLBACK_LEAF_ID = -1

def evalCondition(attributes: Dict[str, Any], condition: Dict[str, Any], savedGroups: Optional[Dict[str, Any]] = None) -> bool:
    for key, value in condition.items():
        if key == "$or":
            if not evalOr(attributes, value, savedGroups):
                return False
        elif key == "$nor":
            if evalOr(attributes, value, savedGroups):
                return False
        elif key == "$and":
            if not evalAnd(attributes, value, savedGroups):
                return False
        elif key == "$not":
            if evalCondition(attributes, value, savedGroups):
                return False
        elif not evalConditionValue(value, getPath(attributes, key), savedGroups):
            return False

    return True

def evalOr(attributes: Dict[str, Any], conditions: List[Any], savedGroups: Optional[Dict[str, Any]]) -> bool:
    if len(conditions) == 0:
        return True

    for condition in conditions:
        if evalCondition(attributes, condition, savedGroups):
            return True
    return False


def evalAnd(attributes: Dict[str, Any], conditions: List[Any], savedGroups: Optional[Dict[str, Any]]) -> bool:
    for condition in conditions:
        if not evalCondition(attributes, condition, savedGroups):
            return False
    return True

def isOperatorObject(obj: Any) -> bool:
    for key in obj.keys():
        if key[0] != "$":
            return False
    return True

def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def getType(attributeValue: Any) -> str:
    if attributeValue is None:
        return "null"
    if isinstance(attributeValue, bool):
        return "boolean"
    if _is_numeric(attributeValue):
        return "number"
    if isinstance(attributeValue, str):
        return "string"
    if isinstance(attributeValue, (list, set)):
        return "array"
    if isinstance(attributeValue, dict):
        return "object"
    return "unknown"

def getPath(attributes: Dict[str, Any], path: str) -> Any:
    current = attributes
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return None
    return current

def evalConditionValue(conditionValue: Any, attributeValue: Any, savedGroups: Optional[Dict[str, Any]], insensitive: bool = False) -> bool:
    if isinstance(conditionValue, dict) and isOperatorObject(conditionValue):
        for key, value in conditionValue.items():
            if not evalOperatorCondition(key, attributeValue, value, savedGroups):
                return False
        return True
    
    # Simple equality comparison with optional case-insensitivity
    if insensitive and isinstance(conditionValue, str) and isinstance(attributeValue, str):
        return conditionValue.lower() == attributeValue.lower()
    
    return bool(conditionValue == attributeValue)

def elemMatch(condition: Dict[str, Any], attributeValue: Any, savedGroups: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(attributeValue, list):
        return False

    for item in attributeValue:
        if isOperatorObject(condition):
            if evalConditionValue(condition, item, savedGroups):
                return True
        else:
            if evalCondition(item, condition, savedGroups):
                return True

    return False

def compare(val1: Any, val2: Any) -> int:
    # IEEE 754: NaN is unordered with everything (including itself), so the
    # "0 if neither > nor <" fallthrough below would wrongly report equal.
    # Raise instead — callers' existing exception handling gives the right
    # truth value: $eq=False, $ne=True, $lt/$lte/$gt/$gte=False.
    if isinstance(val1, float) and math.isnan(val1):
        raise ValueError("NaN")
    if isinstance(val2, float) and math.isnan(val2):
        raise ValueError("NaN")

    if _is_numeric(val1) and not _is_numeric(val2):
        if (val2 is None):
            val2 = 0
        else:
            val2 = float(val2)

    if _is_numeric(val2) and not _is_numeric(val1):
        if (val1 is None):
            val1 = 0
        else:
            val1 = float(val1)

    if val1 > val2:
        return 1
    if val1 < val2:
        return -1
    return 0

def _js_strict_equal(a: Any, b: Any) -> bool:
    """JS === semantics for $eq/$ne. Routed here instead of compare() so
    $lt/$gt keep their JS-aligned coercion via compare() while $eq stays
    strict.

    Three buckets:

    * **Different types** → False (e.g. number 5 vs string "5",
      number 1 vs boolean true).
    * **Container types (array, object)** → False unconditionally.
      JS `===` is reference equality for arrays/objects, and within
      feature evaluation the operator's two operands always come from
      separate JSON parses — different references, never `===`. So in
      the only context this code observes, container $eq must be False.
    * **Primitive same type** (number, string, boolean, null) →
      Python `a == b`. Matches `===` for ints/floats and strings;
      NaN handled correctly because `NaN == NaN` is False in Python.
    """
    ta = getType(a)
    if ta != getType(b):
        return False
    if ta == "array" or ta == "object":
        return False
    return bool(a == b)


def evalOperatorCondition(operator: str, attributeValue: Any, conditionValue: Any, savedGroups: Any) -> bool:
    if operator == "$eq":
        return _js_strict_equal(attributeValue, conditionValue)
    elif operator == "$ne":
        return not _js_strict_equal(attributeValue, conditionValue)
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
        if not isinstance(conditionValue, str):
            return False
        if not conditionValue in savedGroups:
            return False
        return isIn(savedGroups[conditionValue] or [], attributeValue)
    elif operator == "$notInGroup":
        if not isinstance(conditionValue, str):
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
    elif operator == "$regexi":
        try:
            r = re.compile(conditionValue, re.IGNORECASE)
            return bool(r.search(attributeValue))
        except Exception:
            return False
    elif operator == "$notRegex":
        try:
            r = re.compile(conditionValue)
            return not bool(r.search(attributeValue))
        except Exception:
            # Same inverted-default trap as $ne: a missing attribute means
            # re.search(None) raises, but the semantic answer for $notRegex
            # is "the missing value doesn't match the regex" → True.
            return True
    elif operator == "$notRegexi":
        try:
            r = re.compile(conditionValue, re.IGNORECASE)
            return not bool(r.search(attributeValue))
        except Exception:
            return True
    elif operator == "$in":
        if not isinstance(conditionValue, list):
            return False
        return isIn(conditionValue, attributeValue)
    elif operator == "$nin":
        if not isinstance(conditionValue, list):
            return False
        return not isIn(conditionValue, attributeValue)
    elif operator == "$ini":
        if not isinstance(conditionValue, list):
            return False
        return isIn(conditionValue, attributeValue, insensitive=True)
    elif operator == "$nini":
        if not isinstance(conditionValue, list):
            return False
        return not isIn(conditionValue, attributeValue, insensitive=True)
    elif operator == "$elemMatch":
        return elemMatch(conditionValue, attributeValue, savedGroups)
    elif operator == "$size":
        if not isinstance(attributeValue, list):
            return False
        return evalConditionValue(conditionValue, len(attributeValue), savedGroups)
    elif operator == "$all":
        if not isinstance(conditionValue, list):
            return False
        return isInAll(conditionValue, attributeValue, savedGroups, insensitive=False)
    elif operator == "$alli":
        if not isinstance(conditionValue, list):
            return False
        return isInAll(conditionValue, attributeValue, savedGroups, insensitive=True)
    elif operator == "$exists":
        if not conditionValue:
            return attributeValue is None
        return attributeValue is not None
    elif operator == "$type":
        return bool(getType(attributeValue) == conditionValue)
    elif operator == "$not":
        return not evalConditionValue(conditionValue, attributeValue, savedGroups)
    return False

def paddedVersionString(input: Any) -> str:
    # If input is a number, convert to a string
    if _is_numeric(input):
        input = str(input)

    if not input or not isinstance(input, str):
        input = "0"

    return _paddedVersionString(input)


_re_ver_strip = re.compile(r"(^v|\+.*$)")
_re_ver_split = re.compile(r"[-.]")


@lru_cache(maxsize=512)
def _paddedVersionString(input: str) -> str:
    # Remove build info and leading `v` if any
    input = _re_ver_strip.sub("", input)
    # Split version into parts (both core version numbers and pre-release tags)
    # "v1.2.3-rc.1+build123" -> ["1","2","3","rc","1"]
    parts = _re_ver_split.split(input)
    # If it's SemVer without a pre-release, add `~` to the end
    # ["1","0","0"] -> ["1","0","0","~"]
    # "~" is the largest ASCII character, so this will make "1.0.0" greater than "1.0.0-beta" for example
    if len(parts) == 3:
        parts.append("~")
    # Left pad each numeric part with spaces so string comparisons will work ("9">"10", but " 9"<"10")
    # Then, join back together into a single string
    return "-".join([v.rjust(5, " ") if v.isdigit() else v for v in parts])


def isIn(conditionValue: List[Any], attributeValue: Any, insensitive: bool = False) -> bool:
    if insensitive:
        # Helper function to case-fold values (lowercase for strings).
        # Uses Python str.lower(), which is byte-identical to JS toLowerCase()
        # for the relevant inputs: both do Unicode-aware single-char mapping
        # without multi-char folds (e.g., "İ".lower() == "i̇" in both;
        # "ß".lower() == "ß" in both; "Σ".lower() == "σ" in both).
        def case_fold(val: Any) -> Any:
            return val.lower() if isinstance(val, str) else val
        
        # Do an intersection if attribute is an array (insensitive)
        if isinstance(attributeValue, list):
            return any(
                case_fold(el) == case_fold(exp)
                for el in attributeValue
                for exp in conditionValue
            )
        return any(case_fold(attributeValue) == case_fold(exp) for exp in conditionValue)
    
    # Case-sensitive behavior (original)
    if isinstance(attributeValue, list):
        return bool(set(conditionValue) & set(attributeValue))
    return attributeValue in conditionValue

def isInAll(conditionValue: List[Any], attributeValue: Any, savedGroups: Optional[Dict[str, Any]], insensitive: bool = False) -> bool:
    """Check if attributeValue (array) contains all elements in conditionValue"""
    if not isinstance(attributeValue, list):
        return False
    
    for cond in conditionValue:
        passing = False
        for attr in attributeValue:
            if evalConditionValue(cond, attr, savedGroups, insensitive):
                passing = True
                break
        if not passing:
            return False
    return True

def _getOrigHashValue(
    eval_context: EvaluationContext,
    attr: Optional[str] = "id",
    fallbackAttr: Optional[str] = None
) -> Tuple[str, str]:
    # attr = attr or "id" -- Fix for the flaky behavior of sticky bucket assignment
    actual_attr: str = attr if attr is not None else "id"
    val = ""

    if actual_attr in eval_context.user.attributes:
        val = "" if eval_context.user.attributes[actual_attr] is None else eval_context.user.attributes[actual_attr]

    # If no match, try fallback
    if (not val or val == "") and fallbackAttr and eval_context.global_ctx.options.sticky_bucket_service:
        if fallbackAttr in eval_context.user.attributes:
            val = "" if eval_context.user.attributes[fallbackAttr] is None else eval_context.user.attributes[fallbackAttr]

        if val:
            actual_attr = fallbackAttr

    return (actual_attr, val)

def _getHashValue(eval_context: EvaluationContext, attr: Optional[str] = None, fallbackAttr: Optional[str] = None) -> Tuple[str, str]:
    (attr, val) = _getOrigHashValue(attr=attr, fallbackAttr=fallbackAttr, eval_context=eval_context)
    return (attr, str(val))

def _isIncludedInRollout(
    seed: str,
    eval_context: EvaluationContext,
    hashAttribute: Optional[str] = None,
    fallbackAttribute: Optional[str] = None,
    range: Optional[Tuple[float, float]] = None,
    coverage: Optional[float] = None,
    hashVersion: Optional[int] = None
) -> bool:
    if coverage is None and range is None:
        return True

    if coverage == 0 and range is None:
        return False

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


def fnv1a32(s: str) -> int:
    hval = 0x811C9DC5
    if s.isascii():
        for b in s.encode():
            hval = ((hval ^ b) * 0x01000193) & 0xFFFFFFFF
    else:
        for ch in s:
            hval = ((hval ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
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

def _urlIsValid(url: Optional[str], pattern: str) -> bool:
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


# Weight vectors whose sum falls outside this tolerance are replaced with
# equal weights at bucketing time.
WEIGHT_SUM_MIN = 0.99
WEIGHT_SUM_MAX = 1.01


def _is_valid_weight_vector(weights: Any, num_variations: int) -> bool:
    """The single weight-vector validity rule, shared by bucketing and every
    contextual bandit path: a list with one finite, non-negative real number
    per variation (booleans excluded), summing to ~1. Anything else is
    replaced with equal weights wherever weights are consumed, so bucket
    ranges can never be inverted and reported bandit propensities always
    describe the vector actually used.

    Deliberately stricter than the JS SDK, which checks only length and sum
    and will bucket on inverted ranges for e.g. [1.2, -0.2] — corrupt for
    assignments and bandit training alike. The divergence exists only for
    invalid payloads the server never produces (the shared conformance
    corpus exercises none of them)."""
    if not isinstance(weights, list) or len(weights) != num_variations:
        return False
    try:
        if not all(
            isinstance(w, (int, float))
            and not isinstance(w, bool)
            and math.isfinite(w)
            and w >= 0
            for w in weights
        ):
            return False
        return WEIGHT_SUM_MIN <= sum(weights) <= WEIGHT_SUM_MAX
    except OverflowError:
        # math.isfinite (and mixed int/float summation) raise for
        # arbitrary-precision ints too large for a float, e.g. 10**1000.
        # Validation must be total over payload data: such vectors are
        # invalid, never a crash.
        return False


def _normalized_weights(numVariations: int, weights: Any) -> List[float]:
    """The weight vector bucketing (and bandit metadata) will actually use:
    the input when it passes _is_valid_weight_vector, equal weights
    otherwise."""
    if _is_valid_weight_vector(weights, numVariations):
        return cast(List[float], weights)
    return getEqualWeights(numVariations)


def getBucketRanges(
    numVariations: int, coverage: float = 1, weights: Optional[List[float]] = None
) -> List[Tuple[float, float]]:
    if coverage < 0:
        coverage = 0
    if coverage > 1:
        coverage = 1
    weights = _normalized_weights(numVariations, weights)

    cumulative: float = 0
    ranges = []
    for w in weights:
        start = cumulative
        cumulative += w
        ranges.append((start, start + coverage * w))

    return ranges

def _fire_rule_tracks(
    rule_tracks: List[Dict[str, Any]],
    eval_context: EvaluationContext,
) -> None:
    """Fire the context's tracking callback for each deferred
    experiment-tracking entry attached to a remote-eval force rule. The proxy
    server evaluates experiments server-side and emits the resulting
    (experiment, result) pairs here so the SDK can still drive its tracking
    pipeline. Mirrors the JS SDK behavior in packages/sdk-js/src/core.ts
    (`if (rule.tracks) ...`)."""
    tracking_cb = eval_context.tracking_cb
    if not rule_tracks or not tracking_cb:
        return
    for entry in rule_tracks:
        exp_data = entry.get("experiment") or {}
        res_data = entry.get("result") or {}
        # Experiment requires at minimum a key and variations list.
        if "key" not in exp_data or "variations" not in exp_data:
            logger.debug("Skipping rule.tracks entry: missing experiment key/variations")
            continue
        # The proxy emits Result in the JS shape: key/name/passthrough flat at
        # the top level. Python's Result takes those via a nested `meta` dict.
        # Re-pack if no explicit `meta` was provided.
        meta: Optional[VariationMeta] = res_data.get("meta")
        if meta is None:
            flat = cast(VariationMeta, {k: res_data[k] for k in ("key", "name", "passthrough") if k in res_data})
            meta = flat or None
        try:
            # Experiment accepts **_ignored, so passing the raw proxy dict is safe.
            experiment = Experiment(**exp_data)
            # Contextual bandit exposure metadata evaluated by the proxy is
            # payload data like any other: hold it to the same validity rules
            # as local evaluation (the JS SDK passes it through verbatim). An
            # invalid leafId or weight vector drops all bandit metadata; an
            # invalid banditVersion drops just that field.
            leaf_id = res_data.get("leafId")
            variation_weights = res_data.get("variationWeights")
            bandit_version = res_data.get("banditVersion")
            if not _is_valid_bandit_id(leaf_id) or not _is_valid_weight_vector(
                variation_weights, len(experiment.variations)
            ):
                leaf_id = variation_weights = bandit_version = None
            elif not _is_valid_bandit_id(bandit_version):
                bandit_version = None
            result = Result(
                variationId=res_data.get("variationId", 0),
                inExperiment=res_data.get("inExperiment", False),
                value=res_data.get("value"),
                hashUsed=res_data.get("hashUsed", False),
                hashAttribute=res_data.get("hashAttribute", "id"),
                hashValue=res_data.get("hashValue", ""),
                featureId=res_data.get("featureId"),
                meta=meta,
                bucket=res_data.get("bucket"),
                stickyBucketUsed=res_data.get("stickyBucketUsed", False),
                leafId=leaf_id,
                variationWeights=variation_weights,
                banditVersion=bandit_version,
            )
            tracking_cb(experiment, result, eval_context.user)
        except Exception:
            logger.exception("Failed to fire rule.tracks tracking event")


def _get_contextual_bandit_leaf(
    contexts: List[ContextualBanditContext],
    evalContext: EvaluationContext,
) -> Optional[ContextualBanditContext]:
    """Return the first leaf whose condition matches the user's attributes.

    Leaf conditions use the regular targeting condition syntax; an empty
    condition matches everyone (the catch-all leaf)."""
    for context in contexts:
        if evalCondition(
            evalContext.user.attributes,
            context.get("condition") or {},
            evalContext.global_ctx.saved_groups,
        ):
            return context
    return None


def _is_valid_bandit_id(value: Any) -> TypeGuard[int]:
    """The validity rule for the payload's bandit identifiers (leafId,
    banditVersion): server-assigned integers, booleans excluded. Invalid
    identifiers are treated as absent, so exposure callbacks never feed
    corrupt leaf/model ids into bandit attribution and training."""
    return isinstance(value, int) and not isinstance(value, bool)


def _build_contextual_bandit_experiment(
    experiment: Experiment[Any],
    contextual_bandit_ref: str,
    feature_id: str,
    evalContext: EvaluationContext,
) -> None:
    """Resolve a rule's contextualBanditRef and substitute the matched leaf's
    weight vector onto the experiment, mirroring the JS SDK.

    Matched leaf: experiment.weights are replaced with the leaf's weights.
    No match / empty contexts / errored selection: bucketing keeps the rule's
    server-computed marginal weights and the fallback leafId -1 is reported.
    Dangling ref: run as a plain experiment with no bandit metadata at all."""
    # Typed as Any: the map is payload data, so the definition shape is only
    # a promise — the guards below must survive malformed entries at runtime.
    cb_definition: Any = evalContext.global_ctx.contextual_bandits.get(contextual_bandit_ref)
    # An empty-dict definition counts as found (JS `!cbDefinition` semantics):
    # it takes the fallback-leaf path below, not the dangling-ref return.
    if not cb_definition and not isinstance(cb_definition, dict):
        # debug, not warning: this fires on EVERY evaluation of the feature,
        # and a payload-skew window makes it reachable in normal operation.
        # The JS SDK logs these only in debug mode for the same reason.
        logger.debug(
            "Contextual bandit %s not found in payload, feature %s falls back to aggregate weights",
            contextual_bandit_ref,
            feature_id,
        )
        return
    if not isinstance(cb_definition, dict):
        # Malformed payload entry: treat like a definition with no leaves so
        # bucketing still degrades to the rule's aggregate weights instead of
        # crashing the evaluation.
        cb_definition = {}

    leaf = None
    contexts = cb_definition.get("contexts") or []
    if contexts:
        try:
            leaf = _get_contextual_bandit_leaf(contexts, evalContext)
        except Exception:
            logger.debug(
                "Contextual bandit leaf selection failed, feature %s falls back to aggregate weights",
                feature_id,
                exc_info=True,
            )

    weights = leaf.get("weights") if leaf is not None else None
    leaf_id = leaf.get("leafId") if leaf is not None else None
    if weights is not None and not _is_valid_weight_vector(weights, len(experiment.variations)):
        weights = None
    if leaf_id is not None and not _is_valid_bandit_id(leaf_id):
        leaf_id = None
    if leaf is not None and (weights is None or leaf_id is None):
        # A matched leaf missing its id (or carrying a non-integer one), or
        # whose weight vector fails the shared validity rule, is a malformed
        # payload; degrade to the aggregate-weights fallback rather than
        # reporting propensities that differ from the weights actually used.
        logger.debug(
            "Contextual bandit leaf is malformed, feature %s falls back to aggregate weights",
            feature_id,
        )

    if weights is not None and leaf_id is not None:
        experiment.weights = weights
        cb: ContextualBanditAssignment = {"leafId": leaf_id, "variationWeights": weights}
    else:
        logger.debug(
            "Contextual bandit: no matching leaf, feature %s uses aggregate weights", feature_id
        )
        cb = {
            "leafId": CONTEXTUAL_BANDIT_FALLBACK_LEAF_ID,
            # getBucketRanges applies the same normalization, so the reported
            # propensities always match the vector bucketing will use.
            "variationWeights": _normalized_weights(
                len(experiment.variations), experiment.weights
            ),
        }

    bandit_version = cb_definition.get("banditVersion")
    if _is_valid_bandit_id(bandit_version):
        cb["banditVersion"] = bandit_version
    experiment.contextualBandit = cb


def eval_feature(
    key: str,
    evalContext: Optional[EvaluationContext] = None,
) -> FeatureResult[Any]:
    """Core feature evaluation logic as a standalone function.

    Tracking and subscription callbacks are read from the EvaluationContext
    so recursive evaluations (prerequisites) report through them too."""

    if evalContext is None:
        raise ValueError("evalContext is required - eval_feature")
    
    if key not in evalContext.global_ctx.features:
        logger.warning("Unknown feature %s", key)
        return FeatureResult(None, "unknownFeature")

    if key in evalContext.stack.evaluated_features:
        logger.warning("Cyclic prerequisite detected, stack: %s", evalContext.stack.evaluated_features)
        return FeatureResult(None, "cyclicPrerequisite")
 
    evalContext.stack.evaluated_features.add(key)

    feature = evalContext.global_ctx.features[key]

    evaluated_features = evalContext.stack.evaluated_features.copy()

    for rule in feature.rules:
        # Reset the stack for each rule
        evalContext.stack.evaluated_features = evaluated_features.copy()

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
            # Fire deferred experiment tracking events attached by the
            # remote-eval proxy (no-op when the rule was not produced by remote
            # evaluation).
            if rule.tracks:
                _fire_rule_tracks(rule.tracks, evalContext)
            return FeatureResult(rule.force, "force", ruleId=rule.id)

        # Contextual bandit rules carry their variations under
        # contextualVariations; a rule with neither is skipped (this is what
        # lets bandit-unaware SDKs degrade to the default value).
        rule_variations = (
            rule.contextualVariations
            if rule.contextualVariations is not None
            else rule.variations
        )
        if rule_variations is None:
            logger.warning("Skip invalid rule, feature %s", key)
            continue

        exp = Experiment(
            key=rule.key or key,
            variations=rule_variations,
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

        if rule.contextualBanditRef:
            _build_contextual_bandit_experiment(exp, rule.contextualBanditRef, key, evalContext)

        result = run_experiment(experiment=exp, featureId=key, evalContext=evalContext)

        # Bandit metadata is only meaningful for real hashed exposures; strip
        # it from the experiment for forced/QA/coverage-miss outcomes so it
        # doesn't leak into the returned FeatureResult.
        if exp.contextualBandit is not None and not (result.hashUsed and result.inExperiment):
            exp.contextualBandit = None

        if evalContext.callback_subscription:
            evalContext.callback_subscription(exp, result)

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

def eval_prereqs(parentConditions: List[Dict[str, Any]], evalContext: EvaluationContext) -> str:
    evaluated_features = evalContext.stack.evaluated_features.copy()

    for parentCondition in parentConditions:
        # Reset the stack in each iteration
        evalContext.stack.evaluated_features = evaluated_features.copy()

        parent_id = parentCondition.get("id")
        if parent_id is None:
            continue  # Skip if no valid ID
            
        parentRes = eval_feature(key=parent_id, evalContext=evalContext)

        if parentRes.source == "cyclicPrerequisite":
            return "cyclic"

        parent_condition = parentCondition.get("condition")
        if parent_condition is None:
            continue  # Skip if no valid condition
            
        if not evalCondition({'value': parentRes.value}, parent_condition, evalContext.global_ctx.saved_groups):
            if parentCondition.get("gate", False):
                return "gate"
            return "fail"
    return "pass"

def _get_sticky_bucket_experiment_key(experiment_key: str, bucket_version: int = 0) -> str:
    return experiment_key + "__" + str(bucket_version)
    
def _get_sticky_bucket_assignments(evalContext: EvaluationContext,
                                    attr: Optional[str] = None,
                                    fallback: Optional[str] = None) -> Dict[str, str]:
    merged: Dict[str, str] = {}

    # Search for docs stored for attribute(id)
    resolved_attr, hashValue = _getHashValue(attr=attr, eval_context=evalContext)
    key = f"{resolved_attr}||{hashValue}"
    if key in evalContext.user.sticky_bucket_assignment_docs:
        merged = evalContext.user.sticky_bucket_assignment_docs[key].get("assignments", {})

    # Search for docs stored for fallback attribute
    if fallback:
        _, hashValue = _getHashValue(attr=fallback, eval_context=evalContext)
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
    evalContext: EvaluationContext,
    bucket_version: Optional[int] = None,
    min_bucket_version: Optional[int] = None,
    meta: Optional[List[VariationMeta]] = None,
    hash_attribute: Optional[str] = None,
    fallback_attribute: Optional[str] = None,
) -> Dict[str, Any]:
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

# NOTE: both clients' public run() declare `Experiment[T] -> Result[T]` and
# rely on this returning the experiment's own variation value (Result[Any] is
# an unchecked cast at that seam). If a refactor ever makes this return a
# value that isn't one of experiment.variations, the public inference lies.
def run_experiment(experiment: Experiment[Any],
                   featureId: Optional[str] = None,
                   evalContext: Optional[EvaluationContext] = None,
                ) -> Result[Any]:
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
    # Explicit bucket ranges take precedence over weights entirely (step 9),
    # so a contextual bandit experiment carrying ranges has no truthful
    # propensity vector to report: drop the metadata rather than describe a
    # distribution bucketing ignored. Bucketing itself is untouched (same
    # assignment as the JS SDK); the server never emits ranges on contextual
    # bandit rules, so this only fires on hand-crafted payloads.
    if experiment.contextualBandit and experiment.ranges:
        experiment.contextualBandit = None
    # Keep reported bandit propensities in sync with the weights actually
    # used for bucketing: an override may have replaced them, and
    # getBucketRanges normalizes unusable vectors to equal weights.
    elif experiment.contextualBandit and experiment.weights is not None:
        synced: ContextualBanditAssignment = {
            "leafId": experiment.contextualBandit["leafId"],
            "variationWeights": _normalized_weights(
                len(experiment.variations), experiment.weights
            ),
        }
        if "banditVersion" in experiment.contextualBandit:
            synced["banditVersion"] = experiment.contextualBandit["banditVersion"]
        experiment.contextualBandit = synced
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
            experiment_key=experiment.key,
            bucket_version=experiment.bucketVersion,
            min_bucket_version=experiment.minBucketVersion,
            meta=experiment.meta,
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

    # 12. Exclude if in QA mode (global)
    if evalContext.global_ctx.options.qa_mode:
        logger.debug("Skip experiment %s because of QA Mode", experiment.key)
        return _getExperimentResult(experiment=experiment, featureId=featureId, evalContext=evalContext)

    # 12.1. Exclude if user has skip_all_experiments flag set
    if evalContext.user.skip_all_experiments:
        logger.debug("Skip experiment %s because user has skip_all_experiments flag set", experiment.key)
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
            # Mutate in place, never replace: the dict may be shared with the
            # client's sticky bucket cache, and subsequent evals must see this
            # assignment (read-your-writes while persistence is async).
            evalContext.user.sticky_bucket_assignment_docs[data["key"]] = doc
            if evalContext.save_sticky_bucket_doc:
                # Client-provided persistence hook (the async client schedules
                # the write off the event loop, fire-and-forget).
                evalContext.save_sticky_bucket_doc(doc)
            else:
                result_ = evalContext.global_ctx.options.sticky_bucket_service.save_assignments(doc)
                if inspect.iscoroutine(result_):
                    # Async service without client wiring: never awaited here
                    # (this code is synchronous). Close it to avoid a
                    # RuntimeWarning and surface the misconfiguration.
                    result_.close()
                    logger.error(
                        "Async sticky bucket service requires GrowthBookClient; "
                        "assignment doc was not persisted"
                    )

    # 14. Fire the tracking callback if set. The clients' _track wrappers
    # snapshot the user context (tracking_user_context) before invoking the
    # user's callback, so the logged attributes are exactly the ones used
    # for bucketing; snapshotting there instead of here keeps evals
    # allocation-free when no tracking callback is configured.
    if evalContext.tracking_cb:
        evalContext.tracking_cb(experiment, result, evalContext.user)

    # 15. Return the result
    logger.debug("Assigned variation %d in experiment %s", assigned, experiment.key)
    return result

def _generate_sticky_bucket_assignment_doc(
    attribute_name: str,
    attribute_value: str,
    assignments: Dict[str, str],
    evalContext: EvaluationContext,
) -> Dict[str, Any]:
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
    experiment: Experiment[Any],
    evalContext: EvaluationContext,
    variationId: int = -1,
    hashUsed: bool = False,
    featureId: Optional[str] = None,
    bucket: Optional[float] = None,
    stickyBucketUsed: bool = False
) -> Result[Any]:
    inExperiment = True
    if variationId < 0 or variationId > len(experiment.variations) - 1:
        variationId = 0
        inExperiment = False

    meta = None
    if experiment.meta:
        meta = experiment.meta[variationId]

    (hashAttribute, hashValue) = _getOrigHashValue(attr=experiment.hashAttribute,
                                                    fallbackAttr=experiment.fallbackAttribute,
                                                    eval_context=evalContext)

    # Contextual bandit exposure metadata is only reported for real hashed
    # assignments — never for forced variations, QA skips, or coverage misses.
    leaf_id: Optional[int] = None
    variation_weights: Optional[List[float]] = None
    bandit_version: Optional[int] = None
    cb = experiment.contextualBandit
    if cb and hashUsed and inExperiment:
        leaf_id = cb["leafId"]
        variation_weights = cb["variationWeights"]
        bandit_version = cb.get("banditVersion")

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
        stickyBucketUsed=stickyBucketUsed,
        leafId=leaf_id,
        variationWeights=variation_weights,
        banditVersion=bandit_version
    )
