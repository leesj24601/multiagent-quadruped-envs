import importlib.util
import unittest
from pathlib import Path


def load_evaluate_module():
    module_path = Path(__file__).with_name("evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluateSuccessTest(unittest.TestCase):
    def test_pushball_success_uses_3d_distance_threshold(self):
        evaluate = load_evaluate_module()
        self.assertTrue(evaluate.is_pushball_success([1.0, 2.0, 3.0], [1.1, 2.1, 3.0]))
        self.assertFalse(evaluate.is_pushball_success([1.0, 2.0, 3.0], [1.3, 2.0, 3.0]))

    def test_gate_success_uses_any_agent_past_x_threshold(self):
        evaluate = load_evaluate_module()
        self.assertTrue(evaluate.is_gatewithbutton_success([[3.0, 0.0, 0.4], [4.1, 0.0, 0.4]]))
        self.assertFalse(evaluate.is_gatewithbutton_success([[3.9, 0.0, 0.4], [4.0, 0.0, 0.4]]))


if __name__ == "__main__":
    unittest.main()
