import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
import joblib # save my trained model & scaled data 
import os 


class HousePricePrdiction:
    def __init__(self, file_path, target_column="Price"):
        self.file_path = file_path
        self.target_column = target_column
        self.scaler = StandardScaler() 

    
    def load_data(self): 
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
            return data
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    def preprocess_data(self, data):        
        # Check if the target column exists in the data
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the data.")
        

        data.fillna(data.median(), inplace=True)  # Fill missing values with the mean of each column


        # Split the data into features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Split the data into training and testing sets
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])  # Scale numerical features

        return X, y 


    def save_preprocessor(self, file_name="../models/scaler.pkl"): 
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        joblib.dump(self.scaler, file_name)
        print(f"Preprocessor saved to {file_name}") 


if __name__ == "__main__":  
    pipeline = HousePricePrdiction(file_path="/Users/preetyrai/MLOPS_HousePricePrediction/Data/house_price.csv")
    df = pipeline.load_data() 

    x, y = pipeline.preprocess_data(df) 
    save_preprocessor = pipeline.save_preprocessor()







