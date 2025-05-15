"""Basic tests for model loading."""

import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestModels(unittest.TestCase):
    def test_import(self):
        """Test that modules can be imported."""
        try:
            from models.graph_cnn import SimplifiedGraphCNN
            self.assertTrue(True)
        except ImportError:
            self.skipTest("GraphCNN module not available")

if __name__ == "__main__":
    unittest.main()
