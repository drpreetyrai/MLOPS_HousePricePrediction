import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
import joblib # save my trained model & scaled data  
import os 
from sklearn.metrics import mean_squared_error, r2_score 


class HousePricePrediction: 
    def __init__(self, data_path="../data/house_price.csv", model_path="/Users/preetyrai/MLOPS_HousePricePrediction/models/house_price_model.pkl"):  
      self.data_path = data_path 
      self.model_path = model_path 
    
    def load_data(self): 
       df = pd.read_csv(self.data_path) 
       df.dropna(inplace=True) 

       X = df.drop(columns=["Price"]) 
       y = df["Price"]
    
       return X, y
    
    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor() 
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        mse = mean_squared_error(y_test, y_pred) 
        r2 = r2_score(y_test, y_pred) 
        print(f"Model trained with MSE: {mse}, R2: {r2}") 
        return model
    
    def save_model(self, model):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path) 
        print(f"Model saved to {self.model_path}")

    
if __name__ == "__main__":
   house_model = HousePricePrediction() 
   X, y = house_model.load_data() 
   trained_model = house_model.train_model(X, y) 
   house_model.save_model(trained_model)


   
    
