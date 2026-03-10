"""Tests for case-insensitive membership operators: $ini, $nini, $alli"""
import pytest
from growthbook.core import evalCondition


class TestCaseInsensitiveOperators:
    """Test case-insensitive membership operators"""

    def test_ini_pass_case_insensitive_match(self):
        """$ini operator should match case-insensitively"""
        condition = {"name": {"$ini": ["JOHN", "JANE"]}}
        attributes = {"name": "john"}
        assert evalCondition(attributes, condition) is True

    def test_ini_pass_uppercase_value_lowercase_pattern(self):
        """$ini operator should match with uppercase value and lowercase pattern"""
        condition = {"name": {"$ini": ["john", "jane"]}}
        attributes = {"name": "JOHN"}
        assert evalCondition(attributes, condition) is True

    def test_ini_pass_mixed_case(self):
        """$ini operator should match with mixed case"""
        condition = {"name": {"$ini": ["JoHn", "jAnE"]}}
        attributes = {"name": "john"}
        assert evalCondition(attributes, condition) is True

    def test_ini_fail(self):
        """$ini operator should fail when value not in list"""
        condition = {"name": {"$ini": ["JOHN", "JANE"]}}
        attributes = {"name": "bob"}
        assert evalCondition(attributes, condition) is False

    def test_ini_array_pass_1(self):
        """$ini operator should work with array attributes - intersection"""
        condition = {"tags": {"$ini": ["A", "B"]}}
        attributes = {"tags": ["a", "c"]}
        assert evalCondition(attributes, condition) is True

    def test_ini_array_pass_2(self):
        """$ini operator should work with array attributes - multiple matches"""
        condition = {"tags": {"$ini": ["A", "B"]}}
        attributes = {"tags": ["a", "b"]}
        assert evalCondition(attributes, condition) is True

    def test_ini_array_fail(self):
        """$ini operator should fail when no intersection"""
        condition = {"tags": {"$ini": ["A", "B"]}}
        attributes = {"tags": ["c", "d"]}
        assert evalCondition(attributes, condition) is False

    def test_ini_not_array(self):
        """$ini operator should fail when condition value is not an array"""
        condition = {"name": {"$ini": "JOHN"}}
        attributes = {"name": "john"}
        assert evalCondition(attributes, condition) is False

    def test_nini_pass(self):
        """$nini operator should pass when value not in list"""
        condition = {"name": {"$nini": ["JOHN", "JANE"]}}
        attributes = {"name": "bob"}
        assert evalCondition(attributes, condition) is True

    def test_nini_fail_case_insensitive_match(self):
        """$nini operator should fail on case-insensitive match"""
        condition = {"name": {"$nini": ["JOHN", "JANE"]}}
        attributes = {"name": "john"}
        assert evalCondition(attributes, condition) is False

    def test_nini_array_pass(self):
        """$nini operator should pass with array when no intersection"""
        condition = {"tags": {"$nini": ["A", "B"]}}
        attributes = {"tags": ["c", "d"]}
        assert evalCondition(attributes, condition) is True

    def test_nini_array_fail(self):
        """$nini operator should fail with array when there's intersection"""
        condition = {"tags": {"$nini": ["A", "B"]}}
        attributes = {"tags": ["a", "c"]}
        assert evalCondition(attributes, condition) is False

    def test_alli_pass(self):
        """$alli operator should pass when all values match case-insensitively"""
        condition = {"tags": {"$alli": ["A", "B"]}}
        attributes = {"tags": ["a", "b", "c"]}
        assert evalCondition(attributes, condition) is True

    def test_alli_pass_exact_match(self):
        """$alli operator should pass with exact case-insensitive match"""
        condition = {"tags": {"$alli": ["A", "B"]}}
        attributes = {"tags": ["A", "B"]}
        assert evalCondition(attributes, condition) is True

    def test_alli_fail_missing_value(self):
        """$alli operator should fail when not all values present"""
        condition = {"tags": {"$alli": ["A", "B", "C"]}}
        attributes = {"tags": ["a", "b"]}
        assert evalCondition(attributes, condition) is False

    def test_alli_fail_not_array(self):
        """$alli operator should fail when attribute is not an array"""
        condition = {"tags": {"$alli": ["A", "B"]}}
        attributes = {"tags": "a"}
        assert evalCondition(attributes, condition) is False

    def test_alli_pass_with_numbers(self):
        """$alli operator should work with non-string values"""
        condition = {"ids": {"$alli": [1, 2]}}
        attributes = {"ids": [1, 2, 3]}
        assert evalCondition(attributes, condition) is True

    def test_ini_pass_with_numbers(self):
        """$ini operator should work with non-string values"""
        condition = {"id": {"$ini": [1, 2, 3]}}
        attributes = {"id": 2}
        assert evalCondition(attributes, condition) is True

    def test_nini_pass_with_numbers(self):
        """$nini operator should work with non-string values"""
        condition = {"id": {"$nini": [1, 2, 3]}}
        attributes = {"id": 4}
        assert evalCondition(attributes, condition) is True

    def test_case_sensitive_in_vs_case_insensitive_ini(self):
        """$in should be case-sensitive while $ini is case-insensitive"""
        # $in should be case-sensitive (fail)
        condition_in = {"name": {"$in": ["JOHN", "JANE"]}}
        attributes = {"name": "john"}
        assert evalCondition(attributes, condition_in) is False

        # $ini should be case-insensitive (pass)
        condition_ini = {"name": {"$ini": ["JOHN", "JANE"]}}
        assert evalCondition(attributes, condition_ini) is True

    def test_case_sensitive_nin_vs_case_insensitive_nini(self):
        """$nin should be case-sensitive while $nini is case-insensitive"""
        # $nin should be case-sensitive (pass because "john" != "JOHN")
        condition_nin = {"name": {"$nin": ["JOHN", "JANE"]}}
        attributes = {"name": "john"}
        assert evalCondition(attributes, condition_nin) is True

        # $nini should be case-insensitive (fail because "john" matches "JOHN")
        condition_nini = {"name": {"$nini": ["JOHN", "JANE"]}}
        assert evalCondition(attributes, condition_nini) is False

    def test_case_sensitive_all_vs_case_insensitive_alli(self):
        """$all should be case-sensitive while $alli is case-insensitive"""
        # $all should be case-sensitive (fail)
        condition_all = {"tags": {"$all": ["A", "B"]}}
        attributes = {"tags": ["a", "b", "c"]}
        assert evalCondition(attributes, condition_all) is False

        # $alli should be case-insensitive (pass)
        condition_alli = {"tags": {"$alli": ["A", "B"]}}
        assert evalCondition(attributes, condition_alli) is True

    def test_complex_condition_with_ini(self):
        """Complex condition with $ini and $or operators"""
        condition = {
            "$or": [
                {"country": {"$ini": ["USA", "CANADA"]}},
                {"language": {"$ini": ["EN", "FR"]}}
            ]
        }
        
        # Should match on country (case-insensitive)
        attributes1 = {"country": "usa", "language": "de"}
        assert evalCondition(attributes1, condition) is True
        
        # Should match on language (case-insensitive)
        attributes2 = {"country": "germany", "language": "en"}
        assert evalCondition(attributes2, condition) is True
        
        # Should not match
        attributes3 = {"country": "germany", "language": "de"}
        assert evalCondition(attributes3, condition) is False

    def test_complex_condition_with_alli(self):
        """Complex condition with $alli and $and operators"""
        condition = {
            "$and": [
                {"tags": {"$alli": ["PREMIUM", "ACTIVE"]}},
                {"role": {"$ini": ["ADMIN", "MODERATOR"]}}
            ]
        }
        
        # Should match
        attributes1 = {"tags": ["premium", "active", "verified"], "role": "admin"}
        assert evalCondition(attributes1, condition) is True
        
        # Should not match (missing tag)
        attributes2 = {"tags": ["premium"], "role": "admin"}
        assert evalCondition(attributes2, condition) is False
        
        # Should not match (wrong role)
        attributes3 = {"tags": ["premium", "active"], "role": "user"}
        assert evalCondition(attributes3, condition) is False

    def test_empty_array_conditions(self):
        """Test behavior with empty arrays"""
        # Empty condition array should pass (vacuous truth)
        condition_ini_empty = {"tags": {"$ini": []}}
        attributes = {"tags": ["a", "b"]}
        assert evalCondition(attributes, condition_ini_empty) is False
        
        # Empty attribute array should fail for $alli
        condition_alli = {"tags": {"$alli": ["A", "B"]}}
        attributes_empty = {"tags": []}
        assert evalCondition(attributes_empty, condition_alli) is False

    def test_unicode_case_insensitive(self):
        """Test case-insensitive matching with unicode characters"""
        condition = {"name": {"$ini": ["CAFÉ", "RÉSUMÉ"]}}
        attributes = {"name": "café"}
        assert evalCondition(attributes, condition) is True
        
        condition2 = {"name": {"$ini": ["café", "résumé"]}}
        attributes2 = {"name": "CAFÉ"}
        assert evalCondition(attributes2, condition2) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
