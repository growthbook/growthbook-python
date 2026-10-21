"""Plain (non-operator) equality must coerce like the JS SDK's evalConditionValue.

Every expected value was produced by running the JS SDK's evalCondition.
"""

import pytest

from growthbook.core import evalCondition


@pytest.mark.parametrize("name, condition, attributes, expected", [
    ("string condition vs integer attribute", {"x": "10"}, {"x": 10}, True),
    ("string condition vs float attribute", {"x": "1.5"}, {"x": 1.5}, True),
    ("string condition vs whole float attribute", {"x": "10"}, {"x": 10.0}, True),
    ("string condition vs large number uses exponent", {"x": "1e+21"}, {"x": 1e21}, True),
    ("string condition vs small number uses exponent", {"x": "1e-7"}, {"x": 1e-7}, True),
    ("string condition vs boolean true", {"x": "true"}, {"x": True}, True),
    ("string condition vs boolean false", {"x": "false"}, {"x": False}, True),
    ("string True does not match boolean", {"x": "True"}, {"x": True}, False),
    ("string condition vs null attribute", {"x": "null"}, {"x": None}, True),
    ("string condition vs missing attribute", {"x": "null"}, {}, True),
    ("string undefined vs missing attribute", {"x": "undefined"}, {}, False),
    ("empty string vs null attribute", {"x": ""}, {"x": None}, False),
    ("string condition vs array attribute", {"x": "a,b"}, {"x": ["a", "b"]}, True),
    ("string condition vs array with null", {"x": "a,,b"}, {"x": ["a", None, "b"]}, True),
    ("string condition vs object attribute", {"x": "[object Object]"}, {"x": {"a": 1}}, True),
    ("number condition vs numeric string", {"x": 10}, {"x": "10"}, True),
    ("number condition vs padded numeric string", {"x": 10}, {"x": " 10 "}, True),
    ("number condition vs decimal string", {"x": 10}, {"x": "10.0"}, True),
    ("number condition vs hex string", {"x": 31}, {"x": "0x1f"}, True),
    ("number condition vs non-numeric string", {"x": 10}, {"x": "10abc"}, False),
    ("number condition vs inf string", {"x": 10}, {"x": "inf"}, False),
    ("zero condition vs empty string", {"x": 0}, {"x": ""}, True),
    ("number condition vs boolean true", {"x": 1}, {"x": True}, True),
    ("zero condition vs boolean false", {"x": 0}, {"x": False}, True),
    ("zero condition vs null attribute", {"x": 0}, {"x": None}, True),
    ("zero condition vs missing attribute", {"x": 0}, {}, True),
    ("integer condition vs float attribute", {"x": 1}, {"x": 1.0}, True),
    ("float condition vs integer attribute", {"x": 1.5}, {"x": 1}, False),
    ("number condition vs single element array", {"x": 3}, {"x": [3]}, True),
    ("zero condition vs empty array", {"x": 0}, {"x": []}, True),
    ("number condition vs object attribute", {"x": 0}, {"x": {}}, False),
    ("true condition vs string true", {"x": True}, {"x": "true"}, True),
    ("false condition vs string false", {"x": False}, {"x": "false"}, False),
    ("false condition vs string zero", {"x": False}, {"x": "0"}, False),
    ("false condition vs number zero", {"x": False}, {"x": 0}, True),
    ("true condition vs number one", {"x": True}, {"x": 1}, True),
    ("false condition vs empty string", {"x": False}, {"x": ""}, True),
    ("false condition vs missing attribute", {"x": False}, {}, False),
    ("false condition vs null attribute", {"x": False}, {"x": None}, False),
    ("true condition vs empty array", {"x": True}, {"x": []}, True),
    ("true condition vs empty object", {"x": True}, {"x": {}}, True),
    ("null condition vs null attribute", {"x": None}, {"x": None}, True),
    ("null condition vs missing attribute", {"x": None}, {}, True),
    ("null condition vs zero", {"x": None}, {"x": 0}, False),
    ("null condition vs empty string", {"x": None}, {"x": ""}, False),
    ("array condition vs equal array", {"x": [1, "a"]}, {"x": [1, "a"]}, True),
    ("array condition vs whole floats", {"x": [1, 2]}, {"x": [1.0, 2.0]}, True),
    ("array condition vs booleans", {"x": [1, 0]}, {"x": [True, False]}, False),
    ("array condition vs string attribute", {"x": ["a"]}, {"x": "a"}, False),
    ("object condition key order matters", {"x": {"a": 1, "b": 2}}, {"x": {"b": 2, "a": 1}}, False),
    ("object condition integer keys are ordered first", {"x": {"1": 1, "b": 2}}, {"x": {"b": 2, "1": 1}}, True),
    ("object condition vs null attribute", {"x": {"a": 1}}, {"x": None}, False),
    ("$size string condition coerces", {"x": {"$size": "2"}}, {"x": [1, 2]}, True),
])
def test_plain_equality_matches_js(name, condition, attributes, expected):
    assert evalCondition(attributes, condition) is expected
