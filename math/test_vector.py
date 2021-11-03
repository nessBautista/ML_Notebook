import unittest
from vector import Vector

class TestVector(unittest.TestCase):
    def test_init(self):
        v1 = Vector([1,2,3])        
        v2 = Vector([1,2,3])        
        self.assertAlmostEqual(v1,v2)
    
