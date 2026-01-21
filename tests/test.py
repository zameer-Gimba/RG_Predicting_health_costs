import unittest
import pandas as pd

class TestInsuranceDataset(unittest.TestCase):

    def test_dataset_exists_and_loads(self):
        df = pd.read_csv("data/insurance.csv")
        self.assertFalse(df.empty)

    def test_required_columns(self):
        df = pd.read_csv("data/insurance.csv")
        required_columns = {
            "age", "sex", "bmi", "children", "smoker", "region", "expenses"
        }
        self.assertTrue(required_columns.issubset(df.columns))

if __name__ == "__main__":
    unittest.main()
