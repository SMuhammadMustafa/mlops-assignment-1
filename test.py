import unittest
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import numpy as np

class TestModelTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the dataset
        cls.housing = fetch_california_housing()
        cls.X = cls.housing.data
        cls.y = cls.housing.target

    def test_data_loading(self):
        """Test if the dataset is loaded correctly."""
        self.assertEqual(self.X.shape[0], len(self.y), "Features and target should have the same number of rows.")
        self.assertTrue(self.X.shape[1] > 0, "Features should have more than 0 columns.")
        self.assertTrue(len(self.y) > 0, "Target should have more than 0 rows.")

    def test_train_test_split(self):
        """Test if the data splits correctly."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.assertEqual(X_train.shape[0], 0.8 * self.X.shape[0])
        self.assertEqual(X_test.shape[0], 0.2 * self.X.shape[0])
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_model_training(self):
        """Test if the model trains without errors."""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        self.assertTrue(hasattr(model, "predict"), "Model should have a predict method after training.")
        
        # Check if predictions have the correct shape
        predictions = model.predict(X_test)
        self.assertEqual(predictions.shape, (X_test.shape[0],), "Predictions should match the number of test samples.")

    def test_model_saving(self):
        """Test if the model is saved correctly as a pickle file."""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Check if the file was created
        self.assertTrue(os.path.exists('model.pkl'), "Model file should be saved.")


if __name__ == '__main__':
    unittest.main()
