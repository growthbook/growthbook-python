import unittest
from growthbook.core import getPath, evalCondition

class MyDict(dict):
    pass

class TestDictSubclass(unittest.TestCase):
    def test_get_path_with_subclass(self):
        # Test getPath with a dict subclass
        attributes = MyDict({"user": MyDict({"id": "123", "name": "John"})})
        
        self.assertEqual(getPath(attributes, "user.id"), "123")
        self.assertEqual(getPath(attributes, "user.name"), "John")
        self.assertEqual(getPath(attributes, "user.nonexistent"), None)

    def test_eval_condition_with_subclass(self):
        # Test evalCondition with a dict subclass
        attributes = MyDict({"company": "GrowthBook", "meta": MyDict({"plan": "pro"})})
        
        # Simple condition
        condition = {"company": "GrowthBook"}
        self.assertTrue(evalCondition(attributes, condition))
        
        # Nested condition using getPath (indirectly)
        condition = {"meta.plan": "pro"}
        self.assertTrue(evalCondition(attributes, condition))
        
        # Condition failing
        condition = {"meta.plan": "free"}
        self.assertFalse(evalCondition(attributes, condition))

if __name__ == '__main__':
    unittest.main()
