import inspect
import logging
import math
import re
import json
import warnings
from decimal import Decimal
from functools import lru_cache

from urllib.parse import urlparse, parse_qs
from typing import Callable, Optional, Any, Set, Tuple, List, Dict, Union, cast
from typing_extensions import TypeGuard
from .common_types import (
    ContextualBanditAssignment,
    ContextualBanditContext,
    EvaluationContext,
    FeatureResult,
    Experiment,
    Filter,
    PayloadFilters,
    Result,
    UserContext,
    VariationMeta,
)


logger = logging.getLogger("growthbook.core")

# leafId reported when a contextual bandit rule falls back to its marginal
# weights (no leaf condition matched, empty contexts, or leaf selection
# errored). Matches CONTEXTUAL_BANDIT_FALLBACK_LEAF_ID in the JS SDK.
CONTEXTUAL_BANDIT_FALLBACK_LEAF_ID = -1

# True or False, or None when the result depends on an operator or saved group
# type this SDK doesn't support. The public helpers treat None as failing.
CondResult = Optional[bool]

# Keys a condition can have that aren't attributes, besides $not, which is
# also an operator on a value
_CONDITION_OPERATORS = ("$or", "$nor", "$and", "$savedGroup")

# Errors a malformed condition raises while being evaluated
_EVAL_ERRORS = (TypeError, ValueError, AttributeError, IndexError, RecursionError)


def _not(result: CondResult) -> CondResult:
    return None if result is None else not result


def evalCondition(
    attributes: Dict[str, Any], condition: Dict[str, Any],
    savedGroups: Optional[Dict[str, Any]] = None, visited: Optional[Set[str]] = None,
) -> bool:
    try:
        return _eval_condition(attributes, condition, savedGroups, visited) is True
    except _EVAL_ERRORS:
        # Fail the entire condition so $not cannot turn an evaluation error into a match.
        return False


def _eval_condition(
    attributes: Dict[str, Any], condition: Dict[str, Any],
    savedGroups: Optional[Dict[str, Any]], visited: Optional[Set[str]],
) -> CondResult:
    """Evaluate recursively, leaving evaluation error handling to the outer caller."""
    result: CondResult = True
    for key, value in condition.items():
        r: CondResult
        if key == "$or":
            r = _eval_or(attributes, value, savedGroups, visited)
        elif key == "$nor":
            r = _not(_eval_or(attributes, value, savedGroups, visited))
        elif key == "$and":
            r = _eval_and(attributes, value, savedGroups, visited)
        elif key == "$not":
            r = _not(_eval_condition(attributes, value, savedGroups, visited))
        elif key == "$savedGroup":
            r = cast(CondResult, _eval_saved_group(attributes, value, savedGroups, visited))
        elif key == "$savedGroups":
            r = False
        else:
            r = _eval_condition_value(value, getPath(attributes, key), savedGroups, visited=visited)
        # One failing entry decides it, wherever an unsupported one is
        if r is False:
            return False
        if r is None:
            result = None
    return result

def _eval_saved_group(
    attributes: Dict[str, Any], reference: Any, saved_groups: Optional[Dict[str, Any]],
    visited: Optional[Set[str]],
    prune: Optional[Callable[..., Any]] = None,
) -> Any:
    """Resolve a typed saved group, keeping cycle detection local to this branch.

    With `prune`, an undecided list group returns None and a condition group is
    pruned instead of evaluated, so the result may be a condition."""
    if not isinstance(reference, dict):
        return False
    group_id = reference.get("id")
    if not isinstance(group_id, str) or (visited is not None and group_id in visited):
        return False
    if "attributeKey" in reference and not isinstance(reference["attributeKey"], str):
        return False

    entry = saved_groups.get(group_id) if saved_groups is not None else None
    if not isinstance(entry, dict):
        return False

    if entry.get("type") == "list":
        key = reference.get("attributeKey", entry.get("attributeKey"))
        values = entry.get("values")
        if not isinstance(key, str) or not isinstance(values, list):
            return False
        if prune is not None and not _is_known(attributes, key):
            return None
        return isIn(values, getPath(attributes, key))
    if entry.get("type") == "condition":
        condition = entry.get("condition")
        if not isinstance(condition, dict):
            return False
        # Allocate the branch guard only when entering a condition group.
        next_visited = visited | {group_id} if visited else {group_id}
        if prune is not None:
            return prune(attributes, condition, saved_groups, next_visited)
        return _eval_condition(attributes, condition, saved_groups, next_visited)
    # A group type added after this SDK was built. The client SDK may support it.
    return None


def _saved_group_values(saved_groups: Optional[Dict[str, Any]], group_id: str) -> Optional[List[Any]]:
    """Distinguish an absent group (empty list) from a malformed entry (None)."""
    if saved_groups is None or group_id not in saved_groups:
        return []
    entry = saved_groups[group_id]
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict) and entry.get("type") == "list":
        values = entry.get("values")
        if isinstance(values, list):
            return values
    return None


def _is_new_group_type(entry: Any) -> bool:
    """A saved group type added after this SDK was built"""
    return isinstance(entry, dict) and entry.get("type") not in ("list", "condition")


def evalOr(
    attributes: Dict[str, Any], conditions: List[Any], savedGroups: Optional[Dict[str, Any]],
    visited: Optional[Set[str]] = None,
) -> bool:
    return _eval_or(attributes, conditions, savedGroups, visited) is True


def _eval_or(
    attributes: Dict[str, Any], conditions: List[Any], savedGroups: Optional[Dict[str, Any]],
    visited: Optional[Set[str]],
) -> CondResult:
    if len(conditions) == 0:
        return True

    result: CondResult = False
    for condition in conditions:
        r = _eval_condition(attributes, condition, savedGroups, visited)
        if r is True:
            return True
        if r is None:
            result = None
    return result


def evalAnd(
    attributes: Dict[str, Any], conditions: List[Any], savedGroups: Optional[Dict[str, Any]],
    visited: Optional[Set[str]] = None,
) -> bool:
    return _eval_and(attributes, conditions, savedGroups, visited) is True


def _eval_and(
    attributes: Dict[str, Any], conditions: List[Any], savedGroups: Optional[Dict[str, Any]],
    visited: Optional[Set[str]],
) -> CondResult:
    result: CondResult = True
    for condition in conditions:
        r = _eval_condition(attributes, condition, savedGroups, visited)
        if r is False:
            return False
        if r is None:
            result = None
    return result


def pruneCondition(
    attributes: Dict[str, Any], condition: Dict[str, Any],
    savedGroups: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], bool]:
    """Partially evaluate a condition when only some attributes are known, e.g.
    on a server. Attributes missing from `attributes` are unknown. Returns a
    boolean once the result can't depend on them, otherwise the condition minus
    every branch that is already decided."""
    return _prune_with_context(attributes, condition, savedGroups, None)


def _prune_saved_group(
    attributes: Dict[str, Any], reference: Any, saved_groups: Dict[str, Any],
) -> Any:
    """Prune the saved group a reference points to. Returns its result once
    decided, otherwise its condition minus every branch that is already decided."""
    return _eval_saved_group(attributes, reference, saved_groups, None, prune=_prune_with_context)


def _prune_with_context(
    attributes: Dict[str, Any], condition: Dict[str, Any],
    saved_groups: Optional[Dict[str, Any]], visited: Optional[Set[str]],
) -> Union[Dict[str, Any], bool]:
    # One that raises, such as a malformed one, stays as written. Evaluating it
    # fails as a whole, so no part of it can be decided.
    try:
        return _prune_entries(attributes, condition, saved_groups, visited)
    except _EVAL_ERRORS:
        return condition


def _prune_entries(
    attributes: Dict[str, Any], condition: Dict[str, Any],
    saved_groups: Optional[Dict[str, Any]], visited: Optional[Set[str]],
) -> Union[Dict[str, Any], bool]:
    out: Dict[str, Any] = {}
    for key, value in condition.items():
        r: Any
        if key in ("$or", "$nor", "$and"):
            r = _prune_list(attributes, value, saved_groups, visited, key != "$and")
        elif key == "$not":
            r = _prune_entries(attributes, value, saved_groups, visited)
        elif key == "$savedGroup":
            r = _eval_saved_group(attributes, value, saved_groups, visited, prune=_prune_entries)
            # Keep an undecided reference as is rather than inlining the group
            if not isinstance(r, bool):
                r = None
        elif key == "$savedGroups":
            r = False
        else:
            # None when the attribute is unknown, or the value uses an operator
            # this SDK doesn't support, which the client's SDK may
            r = (
                _eval_condition_value(value, getPath(attributes, key), saved_groups, visited=visited)
                if _is_known(attributes, key)
                else None
            )
        # $not and $nor pass when what they hold fails
        if key in ("$not", "$nor") and isinstance(r, bool):
            r = not r

        # One failing entry fails the whole condition, and a passing one is dropped
        if r is False:
            return False
        # None is an undecided leaf, which stays as written
        if r is None:
            out[key] = value
        elif r is not True:
            out[key] = r
    return out if out else True


def _prune_list(
    attributes: Dict[str, Any], conditions: List[Any],
    saved_groups: Optional[Dict[str, Any]], visited: Optional[Set[str]], is_or: bool,
) -> Union[List[Any], bool]:
    """Prune an $or (or an $and) list. Returns the conditions still undecided,
    or the list's result once there are none."""
    out: List[Any] = []
    for condition in conditions:
        r = _prune_entries(attributes, condition, saved_groups, visited)
        # One passing condition decides an $or, and one failing condition an $and
        if is_or and r is True:
            return True
        if not is_or and r is False:
            return False
        if not isinstance(r, bool):
            out.append(r)
    if out:
        return out
    # Every condition failed an $or, or passed an $and. Like evalOr, an empty
    # $or passes.
    return len(conditions) == 0 if is_or else True


def _is_known(attributes: Any, path: str) -> bool:
    """A nested path is unknown, since the server may only have part of the object"""
    return "." not in path and isinstance(attributes, dict) and path in attributes


def isOperatorObject(obj: Any) -> bool:
    # An empty object has no operators, so it passes
    return isinstance(obj, dict) and all(key.startswith("$") for key in obj.keys())

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
    """Read a dotted path. Like JS, only own keys count, so a list exposes its
    canonical indices ("0", "1", ...) and "length"."""
    current: Any = attributes
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        elif isinstance(current, list) and segment == "length":
            current = len(current)
        elif isinstance(current, list) and _is_array_index(segment, len(current)):
            current = current[int(segment)]
        else:
            return None
    return current

def _is_array_index(key: str, length: int) -> bool:
    """Whether `key` is a canonical index below `length`, such as "0" but not "01" or "-0"."""
    return key.isascii() and key.isdigit() and (key == "0" or key[0] != "0") and int(key) < length

def evalConditionValue(
    conditionValue: Any, attributeValue: Any, savedGroups: Optional[Dict[str, Any]],
    insensitive: bool = False, visited: Optional[Set[str]] = None,
) -> bool:
    return _eval_condition_value(conditionValue, attributeValue, savedGroups, insensitive, visited) is True

def _eval_condition_value(
    conditionValue: Any, attributeValue: Any, savedGroups: Optional[Dict[str, Any]],
    insensitive: bool = False, visited: Optional[Set[str]] = None,
) -> CondResult:
    if isOperatorObject(conditionValue):
        # This is a special operator condition and we should evaluate each one separately
        result: CondResult = True
        for key, value in conditionValue.items():
            r = _eval_operator_condition(key, attributeValue, value, savedGroups, visited)
            if r is False:
                return False
            if r is None:
                result = None
        return result

    # Simple equality comparison: values of different types never match
    if insensitive and isinstance(conditionValue, str) and isinstance(attributeValue, str):
        return conditionValue.lower() == attributeValue.lower()

    return _strict_deep_equal(conditionValue, attributeValue)

def _strict_deep_equal(a: Any, b: Any) -> bool:
    """Literal equality, matching JS, which compares the JSON.stringify output
    of objects and arrays. Every nested value must have the same JSON type, so
    True never equals 1 the way it does with Python's ==, object keys must be
    in the same order, and 1.0 equals 1."""
    ta = getType(a)
    if ta != getType(b):
        return False
    if ta == "array":
        return len(a) == len(b) and all(_strict_deep_equal(_json_value(x), _json_value(y)) for x, y in zip(a, b))
    if ta == "object":
        return (
            list(a.keys()) == list(b.keys())
            and all(_strict_deep_equal(_json_value(a[k]), _json_value(b[k])) for k in a)
        )
    return bool(a == b)

def _json_value(v: Any) -> Any:
    """JSON.stringify writes NaN and Infinity as null"""
    return None if isinstance(v, float) and not math.isfinite(v) else v

def _is_orderable(a: Any, b: Any) -> bool:
    """Only two numbers or two strings can be ordered. Booleans are not ordered."""
    return (_is_numeric(a) and _is_numeric(b)) or (isinstance(a, str) and isinstance(b, str))

def elemMatch(
    condition: Dict[str, Any], attributeValue: Any, savedGroups: Optional[Dict[str, Any]],
    visited: Optional[Set[str]] = None,
) -> bool:
    return _elem_match(condition, attributeValue, savedGroups, visited) is True

def _elem_match(
    condition: Dict[str, Any], attributeValue: Any, savedGroups: Optional[Dict[str, Any]],
    visited: Optional[Set[str]],
) -> CondResult:
    if not isinstance(attributeValue, list):
        return False

    # Operators test each element itself. A condition, including one that only
    # uses $or, $and, $nor, or $savedGroup, tests each element's fields.
    on_value = isOperatorObject(condition) and not any(k in _CONDITION_OPERATORS for k in condition)
    result: CondResult = False
    for item in attributeValue:
        # Only skip None elements; falsy values like 0, "" and False are valid
        # array members and must still be tested against the condition.
        if item is None:
            continue
        if on_value:
            r = _eval_condition_value(condition, item, savedGroups, visited=visited)
        else:
            r = _eval_condition(item, condition, savedGroups, visited)
        if r is True:
            return True
        if r is None:
            result = None
    return result

def compare(val1: Any, val2: Any) -> int:
    # IEEE 754: NaN is unordered with everything (including itself), so the
    # "0 if neither > nor <" fallthrough below would wrongly report equal.
    # Raise instead — callers' existing exception handling gives the right
    # truth value: $eq=False, $ne=True, $lt/$lte/$gt/$gte=False.
    if isinstance(val1, float) and math.isnan(val1):
        raise ValueError("NaN")
    if isinstance(val2, float) and math.isnan(val2):
        raise ValueError("NaN")

    if val1 > val2:
        return 1
    if val1 < val2:
        return -1
    return 0

def _js_strict_equal(a: Any, b: Any) -> bool:
    """JS === semantics for $eq/$ne.

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


def evalOperatorCondition(
    operator: str, attributeValue: Any, conditionValue: Any, savedGroups: Any,
    visited: Optional[Set[str]] = None,
) -> bool:
    return _eval_operator_condition(operator, attributeValue, conditionValue, savedGroups, visited) is True

def _eval_operator_condition(
    operator: str, attributeValue: Any, conditionValue: Any, savedGroups: Any,
    visited: Optional[Set[str]] = None,
) -> CondResult:
    if operator == "$eq":
        return _js_strict_equal(attributeValue, conditionValue)
    elif operator == "$ne":
        return not _js_strict_equal(attributeValue, conditionValue)
    elif operator in ("$lt", "$lte", "$gt", "$gte"):
        if not _is_orderable(attributeValue, conditionValue):
            return False
        try:
            result = compare(attributeValue, conditionValue)
        except ValueError:
            return False
        if operator == "$lt":
            return result < 0
        if operator == "$lte":
            return result <= 0
        if operator == "$gt":
            return result > 0
        return result >= 0
    elif operator in ("$veq", "$vne", "$vlt", "$vlte", "$vgt", "$vgte"):
        # $vne is the negation of $veq, so it matches on a type mismatch
        if not _is_orderable(attributeValue, conditionValue):
            return operator == "$vne"
        actual = paddedVersionString(attributeValue)
        expected = paddedVersionString(conditionValue)
        if operator == "$veq":
            return actual == expected
        if operator == "$vne":
            return actual != expected
        if operator == "$vlt":
            return actual < expected
        if operator == "$vlte":
            return actual <= expected
        if operator == "$vgt":
            return actual > expected
        return actual >= expected
    elif operator in ("$inGroup", "$notInGroup"):
        if not isinstance(conditionValue, str):
            return False
        values = savedGroups.get(conditionValue) if savedGroups is not None else None
        if _is_new_group_type(values):
            return None
        if not isinstance(values, list):
            values = _saved_group_values(savedGroups, conditionValue)
        if values is None:
            return False
        matches = isIn(values, attributeValue)
        return matches if operator == "$inGroup" else not matches
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
        return _elem_match(conditionValue, attributeValue, savedGroups, visited)
    elif operator == "$size":
        if not isinstance(attributeValue, list):
            return False
        return _eval_condition_value(conditionValue, len(attributeValue), savedGroups, visited=visited)
    elif operator == "$all":
        if not isinstance(conditionValue, list):
            return False
        return _is_in_all(conditionValue, attributeValue, savedGroups, insensitive=False, visited=visited)
    elif operator == "$alli":
        if not isinstance(conditionValue, list):
            return False
        return _is_in_all(conditionValue, attributeValue, savedGroups, insensitive=True, visited=visited)
    elif operator == "$exists":
        if not conditionValue:
            return attributeValue is None
        return attributeValue is not None
    elif operator == "$type":
        return bool(getType(attributeValue) == conditionValue)
    elif operator == "$not":
        return _not(_eval_condition_value(conditionValue, attributeValue, savedGroups, visited=visited))
    # Keys this SDK knows, but on a condition rather than on a value, where
    # they never match
    if operator in _CONDITION_OPERATORS:
        return False
    logger.debug("Unknown operator: %s", operator)
    return None

def _js_number_to_string(value: Union[int, float]) -> str:
    """Format a number like JS `String(number)` (ECMAScript Number::toString).

    Integers are converted to doubles first, as JS has only doubles. `repr`
    gives the shortest digits that round-trip, which are then laid out in
    fixed notation for decimal exponents from -6 to 21 and in JS-style
    scientific notation (such as "1e+21" or "1.5e-7") otherwise.
    """
    try:
        x = float(value)
    except OverflowError:
        x = math.inf if value > 0 else -math.inf
    if math.isnan(x):
        return "NaN"
    if x == 0:
        return "0"
    if x < 0:
        return "-" + _js_number_to_string(-x)
    if math.isinf(x):
        return "Infinity"
    # x = 0.d1..dk * 10^n
    _, digit_tuple, exponent = Decimal(repr(x)).normalize().as_tuple()
    digits = "".join(map(str, digit_tuple))
    k = len(digits)
    n = int(exponent) + k
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * -n + digits
    mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
    return mantissa + "e" + ("+" if n > 0 else "-") + str(abs(n - 1))


def paddedVersionString(input: Any) -> str:
    # If input is a number, convert to a string the way JS does
    if _is_numeric(input):
        input = _js_number_to_string(input)

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
                _js_strict_equal(case_fold(el), case_fold(exp))
                for el in attributeValue
                for exp in conditionValue
            )
        return any(
            _js_strict_equal(case_fold(attributeValue), case_fold(exp))
            for exp in conditionValue
        )
    
    if isinstance(attributeValue, list):
        try:
            matches = set(conditionValue) & set(attributeValue)
            # Python collapses booleans with 0/1; any other match is unambiguous.
            if not matches or matches.difference((0, 1)):
                return bool(matches)
            return any(_contains_boolean_or_number(conditionValue, value)
                       for value in attributeValue if value == 0 or value == 1)
        except TypeError:
            # JSON arrays may contain objects or arrays, which cannot be hashed.
            pass
        return any(_js_strict_equal(value, expected)
                   for value in attributeValue for expected in conditionValue)
    # JS includes() compares objects by reference, so an object never matches
    if isinstance(attributeValue, dict):
        return False
    if attributeValue == 0 or attributeValue == 1:
        return _contains_boolean_or_number(conditionValue, attributeValue)
    return attributeValue in conditionValue


def _contains_boolean_or_number(values: List[Any], value: Any) -> bool:
    """Find 0/1 or False/True without Python's boolean/number equality aliasing."""
    start = 0
    while True:
        try:
            index = values.index(value, start)
        except ValueError:
            return False
        if isinstance(values[index], bool) == isinstance(value, bool):
            return True
        start = index + 1

def isInAll(
    conditionValue: List[Any], attributeValue: Any, savedGroups: Optional[Dict[str, Any]],
    insensitive: bool = False, visited: Optional[Set[str]] = None,
) -> bool:
    """Check if attributeValue (array) contains all elements in conditionValue"""
    return _is_in_all(conditionValue, attributeValue, savedGroups, insensitive, visited) is True

def _is_in_all(
    conditionValue: List[Any], attributeValue: Any, savedGroups: Optional[Dict[str, Any]],
    insensitive: bool = False, visited: Optional[Set[str]] = None,
) -> CondResult:
    if not isinstance(attributeValue, list):
        return False

    result: CondResult = True
    for cond in conditionValue:
        passed: CondResult = False
        for attr in attributeValue:
            r = _eval_condition_value(cond, attr, savedGroups, insensitive, visited)
            if r is True:
                passed = True
                break
            if r is None:
                passed = None
        if passed is False:
            return False
        if passed is None:
            result = None
    return result

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
        # Half-open, like inRange: a coverage of k/10000 includes exactly k buckets
        return n < coverage

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
    try:
        res = urlparse(url)
    except ValueError:  # e.g. a malformed host like "https://[::1/"
        return None
    if not res.query:
        return None
    qs = parse_qs(res.query)
    if id not in qs:
        return None
    variation = qs[id][0]
    # str.isdigit() also accepts non-ASCII digits such as "²" (which int()
    # rejects) and "١" (which the JS SDK's parseInt rejects); only 0-9 count.
    if variation is None or not (variation.isascii() and variation.isdigit()):
        return None
    try:
        varId = int(variation)
    except ValueError:  # more digits than Python's int string conversion limit
        return None
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
    """Report each pre-evaluated experiment-tracking entry attached to a
    remote-eval force rule through the context's tracking buffer and tracking
    callback. The proxy server evaluates experiments server-side and emits the
    resulting (experiment, result) pairs here so the SDK can still drive its
    tracking pipeline. Mirrors the JS SDK behavior in
    packages/sdk-js/src/core.ts (`if (rule.tracks) ...`)."""
    if not rule_tracks or (
        eval_context.tracking_cb is None and eval_context.tracking_buffer is None
    ):
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
            _report_exposure(experiment, result, eval_context)
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


def _report_exposure(
    experiment: Experiment[Any],
    result: Result[Any],
    evalContext: EvaluationContext,
) -> None:
    """Report one experiment exposure: deferred tracking buffer first (it
    snapshots for itself, so a failing callback can never lose it), then the
    tracking callback. Buffering is independent of the callback — both fire."""
    if evalContext.tracking_buffer is not None:
        evalContext.tracking_buffer.record(experiment, result, evalContext.user)
    if evalContext.tracking_cb:
        evalContext.tracking_cb(experiment, result, evalContext.user)


def _report_feature_usage(
    key: str,
    result: FeatureResult[Any],
    evalContext: EvaluationContext,
) -> None:
    """Fire the context's feature-usage callback, once per feature key per
    evaluation, however many times a prerequisite chain re-visits it."""
    cb = evalContext.feature_usage_cb
    if cb is None:
        return
    reported = evalContext.reported_features
    if reported is None:
        reported = evalContext.reported_features = set()
    if key in reported:
        return
    reported.add(key)
    cb(key, result, evalContext.user)


def _warn_legacy_callback(name: str) -> None:
    warnings.warn(
        f"the {name} argument is deprecated; set EvaluationContext.{name}",
        DeprecationWarning,
        stacklevel=4,
    )


def eval_feature(
    key: str,
    evalContext: Optional[EvaluationContext] = None,
    callback_subscription: Optional[Callable[[Experiment[Any], Result[Any]], None]] = None,
    tracking_cb: Optional[Callable[[Experiment[Any], Result[Any], UserContext], None]] = None,
) -> FeatureResult[Any]:
    """Core feature evaluation logic as a standalone function.

    Tracking, subscription, and feature-usage callbacks are read from the
    EvaluationContext so recursive evaluations (prerequisites) report through
    them too. The callback keyword arguments are a deprecated,
    invocation-scoped compatibility shim: they are installed on the context
    for the duration of this call (so prerequisites inherit them) and the
    previous context fields are restored afterwards, matching their pre-3.1
    per-call behavior."""

    if evalContext is None:
        raise ValueError("evalContext is required - eval_feature")

    if callback_subscription is None and tracking_cb is None:
        # Inlined _eval_feature_and_report: this is the hot path, and the
        # extra call frame is measurable at millions of evals per second.
        result = _eval_feature(key, evalContext)
        if evalContext.feature_usage_cb is not None:
            _report_feature_usage(key, result, evalContext)
        return result

    previous = (evalContext.callback_subscription, evalContext.tracking_cb)
    if callback_subscription is not None:
        _warn_legacy_callback("callback_subscription")
        evalContext.callback_subscription = callback_subscription
    if tracking_cb is not None:
        _warn_legacy_callback("tracking_cb")
        evalContext.tracking_cb = tracking_cb
    try:
        return _eval_feature_and_report(key, evalContext)
    finally:
        evalContext.callback_subscription, evalContext.tracking_cb = previous


def _eval_feature_and_report(key: str, evalContext: EvaluationContext) -> FeatureResult[Any]:
    result = _eval_feature(key, evalContext)
    if evalContext.feature_usage_cb is not None:
        _report_feature_usage(key, result, evalContext)
    return result


def _eval_feature(
    key: str,
    evalContext: EvaluationContext,
) -> FeatureResult[Any]:

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

        result = _run_experiment(exp, key, evalContext)

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
            
        parentRes = _eval_feature_and_report(parent_id, evalContext)

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
                   tracking_cb: Optional[Callable[[Experiment[Any], Result[Any], UserContext], None]] = None,
                ) -> Result[Any]:
    if evalContext is None:
        raise ValueError("evalContext is required - run_experiment")
    if tracking_cb is None:
        return _run_experiment(experiment, featureId, evalContext)
    # Deprecated, invocation-scoped compatibility shim (see eval_feature).
    _warn_legacy_callback("tracking_cb")
    previous = evalContext.tracking_cb
    evalContext.tracking_cb = tracking_cb
    try:
        return _run_experiment(experiment, featureId, evalContext)
    finally:
        evalContext.tracking_cb = previous


def _run_experiment(experiment: Experiment[Any],
                    featureId: Optional[str],
                    evalContext: EvaluationContext,
                ) -> Result[Any]:
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

    # 14. Report the exposure (see _report_exposure: buffer first, then the
    # tracking callback). The clients' _track wrappers snapshot the user
    # context (tracking_user_context) before invoking the user's callback, so
    # the logged attributes are exactly the ones used for bucketing;
    # snapshotting there instead of here keeps evals allocation-free when no
    # tracking callback is configured.
    _report_exposure(experiment, result, evalContext)

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

    # A meta list shorter than the variations list is a malformed payload:
    # treat the missing entry as absent instead of raising. (The JS SDK
    # crashes here too — reading `.key` off undefined — so this is a
    # deliberate divergence on invalid payloads only.)
    meta = None
    if experiment.meta and variationId < len(experiment.meta):
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

    # The clamp above leaves variationId at 0 even when the variations list is
    # empty (a malformed payload — e.g. `variations: []` or
    # `contextualVariations: []`). The JS SDK reads `undefined` there and the
    # rule degrades to the feature's default; mirror it with None instead of
    # raising IndexError.
    value = experiment.variations[variationId] if experiment.variations else None

    return Result(
        featureId=featureId,
        inExperiment=inExperiment,
        variationId=variationId,
        value=value,
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


def _js_truthy(value: Any) -> bool:
    """JavaScript truthiness, where an empty list or dict counts as true"""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return value != ""
    return True


def _saved_groups_on_cycles(saved_groups: Dict[str, Any]) -> Set[str]:
    """The saved groups that can reach themselves through saved-group
    references: those in a strongly connected component of more than one group,
    or that reference themselves. Each group's references are collected once,
    and Tarjan's algorithm runs without recursion."""
    refs: Dict[str, Set[str]] = {}
    for group_id, entry in saved_groups.items():
        found: Set[str] = set()
        stack: List[Any] = [entry]
        while stack:
            value = stack.pop()
            if isinstance(value, str):
                if value in saved_groups:
                    found.add(value)
            elif isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        refs[group_id] = found

    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    path: List[str] = []
    on_path: Set[str] = set()
    result: Set[str] = set()
    for root in refs:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        path.append(root)
        on_path.add(root)
        work = [(root, iter(refs[root]))]
        while work:
            node, children = work[-1]
            descended = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = len(index)
                    path.append(child)
                    on_path.add(child)
                    work.append((child, iter(refs[child])))
                    descended = True
                    break
                if child in on_path:
                    low[node] = min(low[node], index[child])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: List[str] = []
                while True:
                    member = path.pop()
                    on_path.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or node in refs[node]:
                    result.update(component)
    return result


def reduce_payload(
    payload: Dict[str, Any],
    attributes: Optional[Dict[str, Any]] = None,
    include: Optional[PayloadFilters] = None,
    exclude: Optional[PayloadFilters] = None,
    prerequisites: str = "keep",
) -> Dict[str, Any]:
    """Shrink a decrypted SDK payload for a user whose attributes are only
    partly known, e.g. on a server bootstrapping a client SDK. The result
    evaluates the same as the full payload for anyone who agrees with the known
    attributes.

    `include` keeps only the features and auto experiments that match every
    filter, and `exclude` removes the ones that match any. With
    `prerequisites="keep"` (the default) the prerequisites of what is kept are
    sent even when the filters remove them; `"filter"` applies the filters to
    them too. Mirrors the JS SDK's reducePayload."""
    known: Dict[str, Any] = attributes or {}
    inc: Dict[str, Any] = cast(Dict[str, Any], include or {})
    exc: Dict[str, Any] = cast(Dict[str, Any], exclude or {})
    saved_groups: Dict[str, Any] = payload.get("savedGroups") or {}
    all_features: Dict[str, Any] = payload.get("features") or {}
    all_experiments: List[Any] = payload.get("experiments") or []

    # Stands for a missing key, which is undefined in JS and matches nothing, not even null
    missing = object()

    def has(actual: Any, want: Any) -> bool:
        # Only a list counts, as with $in, and any one of its values is enough
        return isinstance(want, list) and actual is not missing and isIn(want, actual)

    def meta_of(item: Any) -> Dict[str, Any]:
        meta = item.get("metadata") if isinstance(item, dict) else None
        return meta if isinstance(meta, dict) else {}

    def rule_in_projects(meta: Any) -> bool:
        # A rule without projects applies wherever its feature is delivered
        return (
            not _js_truthy(inc.get("projects"))
            or not isinstance(meta, dict)
            or not _js_truthy(meta.get("projects"))
            or has(meta.get("projects"), inc.get("projects"))
        )

    def in_projects(meta: Dict[str, Any]) -> bool:
        # A feature or experiment has to list one of the projects, or be in all
        # of them. One without projects is in none, like one without tags.
        return (
            not _js_truthy(inc.get("projects"))
            or _js_truthy(meta.get("allProjects"))
            or has(meta.get("projects", missing), inc.get("projects"))
        )

    def matches(meta: Dict[str, Any]) -> bool:
        # Kept when it matches every include filter and no exclude filter. Only
        # listed projects are excluded, so one in all projects stays.
        fields = meta.get("customFields")
        fields = fields if isinstance(fields, dict) else {}
        in_fields = inc.get("customFields")
        in_fields = in_fields if isinstance(in_fields, dict) else {}
        out_fields = exc.get("customFields")
        out_fields = out_fields if isinstance(out_fields, dict) else {}
        return (
            in_projects(meta)
            and (not _js_truthy(inc.get("tags")) or has(meta.get("tags", missing), inc.get("tags")))
            and all(has(fields.get(k, missing), v) for k, v in in_fields.items())
            and not has(meta.get("projects", missing), exc.get("projects"))
            and not has(meta.get("tags", missing), exc.get("tags"))
            and not any(has(fields.get(k, missing), v) for k, v in out_fields.items())
        )

    def prune(item: Dict[str, Any], is_rule: bool) -> Dict[str, Any]:
        # Reduces a rule or an auto experiment
        condition: Any = (
            pruneCondition(known, item["condition"], saved_groups)
            if _js_truthy(item.get("condition"))
            else None
        )
        item = dict(item)
        if condition is True:
            del item["condition"]
        # Kept, since sticky buckets and forced variations skip an experiment's
        # condition, and a rule's prerequisites are checked before it
        elif condition is False:
            item["condition"] = {"$not": {}}
            # A force rule that never passes never returns its value. An auto
            # experiment's forced variation still applies to a sticky bucketed user.
            if is_rule and "force" in item:
                item["force"] = None
        elif condition:
            item["condition"] = condition
        return item

    # Adds a feature, then the prerequisites the client evaluates for it, even
    # when the filters drop them. With "filter", only the filters decide,
    # except for holdouts, which aren't app features and are always kept.
    added: Dict[str, Any] = {}

    def add_parents(item: Dict[str, Any]) -> None:
        for parent in item.get("parentConditions") or []:
            parent_id = parent.get("id") if isinstance(parent, dict) else None
            if not isinstance(parent_id, str):
                continue
            if prerequisites != "filter" or parent_id.startswith("$holdout:"):
                add(parent_id)

    def add(feature_id: str) -> None:
        feature = all_features.get(feature_id)
        if not isinstance(feature, dict) or feature_id in added:
            return
        # Rules scoped to other projects aren't served. A prerequisite from
        # another project keeps its own, as on the server.
        scoped = in_projects(meta_of(feature))
        rules: Optional[List[Any]] = None
        if feature.get("rules") is not None:
            rules = [
                prune(rule, True) for rule in feature["rules"]
                if not scoped or rule_in_projects(rule.get("metadata"))
            ]
        # Stored before its parents are added, so a prerequisite cycle ends
        added[feature_id] = {**feature, "rules": rules} if rules is not None else feature
        for rule in rules or []:
            add_parents(rule)

    for feature_id, feature in all_features.items():
        if matches(meta_of(feature)):
            add(feature_id)
    experiments = [prune(e, False) for e in all_experiments if isinstance(e, dict) and matches(meta_of(e))]
    for experiment in experiments:
        add_parents(experiment)
    features = dict(added)

    # The contextual bandits that kept rules and experiments use. A context
    # whose condition never passes can't be the first that does, so it goes.
    all_bandits: Dict[str, Any] = payload.get("contextualBandits") or {}
    bandits: Dict[str, Any] = {}
    items = [rule for f in features.values() for rule in (f.get("rules") or [])] + experiments
    for item in items:
        ref = item.get("contextualBanditRef")
        if not isinstance(ref, str) or not ref or ref in bandits or ref not in all_bandits:
            continue
        bandit = all_bandits[ref]
        # Evaluation tolerates a definition that isn't an object, so it stays
        if not isinstance(bandit, dict):
            bandits[ref] = bandit
            continue
        contexts = []
        for context in bandit.get("contexts") or []:
            pruned = pruneCondition(known, context.get("condition") or {}, saved_groups)
            if pruned is False:
                continue
            contexts.append({**context, "condition": {} if pruned is True else pruned})
        bandits[ref] = {**bandit, "contexts": contexts}

    # Groups that keep their condition as written: those evaluated against
    # something other than the user (each item in $elemMatch, or the parent's
    # value in a prerequisite), and the groups those reference
    # The walks below use explicit stacks, so long chains of groups can't hit
    # the recursion limit.
    as_written: Set[str] = set()
    seen: Set[str] = set()
    marking: List[Tuple[Any, bool]] = [([features, experiments, bandits], True)]
    while marking:
        value, user = marking.pop()
        if isinstance(value, str):
            if value not in saved_groups or value in as_written:
                continue
            if user and value in seen:
                continue
            (seen if user else as_written).add(value)
            marking.append((saved_groups[value], user))
        elif isinstance(value, dict):
            for k, v in value.items():
                marking.append((v, user and k != "$elemMatch" and k != "parentConditions"))
        elif isinstance(value, list):
            for v in value:
                marking.append((v, user))

    # A group on a cycle can't be reduced on its own: its result depends on
    # which group the evaluation entered the cycle from
    on_cycle = _saved_groups_on_cycles(saved_groups)

    # Keep each saved group whose id appears as a string anywhere, with
    # condition groups reduced. This needs no list of the operators that
    # reference groups. Visited depth first, in document order.
    groups: Dict[str, Any] = {}
    keeping: List[Any] = [[features, experiments, bandits]]
    while keeping:
        value = keeping.pop()
        if isinstance(value, str):
            if value not in saved_groups or value in groups:
                continue
            condition = (
                None
                if value in as_written or value in on_cycle
                else _prune_saved_group(known, {"id": value}, saved_groups)
            )
            entry = saved_groups[value]
            groups[value] = (
                {**entry, "condition": condition}
                if isinstance(condition, dict) and isinstance(entry, dict)
                else entry
            )
            keeping.append(groups[value])
        elif isinstance(value, dict):
            keeping.extend(reversed(list(value.values())))
        elif isinstance(value, list):
            keeping.extend(reversed(value))

    reduced: Dict[str, Any] = {
        **payload,
        "features": features,
        "experiments": experiments,
        "savedGroups": groups,
    }
    # Only when there were any, as with the payload's other optional keys
    if isinstance(payload.get("contextualBandits"), dict):
        reduced["contextualBandits"] = bandits
    return reduced
