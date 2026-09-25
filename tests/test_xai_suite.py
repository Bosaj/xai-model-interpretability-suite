import os
import sys
import unittest
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pkg_root = os.path.join(repo_root, "ELHADJI Oussama")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)


class TestXAISuite(unittest.TestCase):
    def test_datasets_exist(self):
        dataset_dir = os.path.join(pkg_root, "Dataset")
        self.assertTrue(os.path.isdir(dataset_dir))
        for filename in ["credit.csv", "fifa.csv", "wheat_seeds.csv"]:
            csv_path = os.path.join(dataset_dir, filename)
            self.assertTrue(os.path.exists(csv_path), f"Dataset {filename} must exist")
            self.assertGreater(os.path.getsize(csv_path), 0)

    def test_notebooks_exist(self):
        for ex in ["Exercice_1.ipynb", "Exercice_2.ipynb", "Exercice_3.ipynb", "Exercice_4.ipynb"]:
            nb_path = os.path.join(pkg_root, ex)
            self.assertTrue(os.path.exists(nb_path), f"Notebook {ex} must exist")

    def test_get_grid_computation(self):
        from utils.lime import get_grid

        class MockDataset:
            def __init__(self):
                self.X = np.array([[0.0, 0.0], [1.0, 1.0]])

        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        u, v, z = get_grid(MockModel(), MockDataset(), points_per_feature=5)
        self.assertEqual(len(u), 5)
        self.assertEqual(len(v), 5)
        self.assertEqual(z.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
