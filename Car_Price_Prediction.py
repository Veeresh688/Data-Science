
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

dataset_dir = kagglehub.dataset_download("vijayaadithyanvg/car-price-predictionused-cars")

files = os.listdir(dataset_dir)
print("Files in dataset folder:", files)

csv_file = os.path.join(dataset_dir, files[0])
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

df = pd.read_csv(csv_file)

X = df.drop(['Car_Name','Selling_Price'], axis=1)
y = df['Selling_Price']

num_col = ['Year', 'Present_Price', 'Driven_kms', 'Owner']
cat_col = ['Fuel_Type', 'Selling_type','Transmission']

column_transformer = ColumnTransformer(transformers=[
('num', StandardScaler(), num_col),
('cat', OneHotEncoder(drop='first'),cat_col)
])

pipeline = Pipeline(steps =
[
('preprocessor', column_transformer),
('model',LinearRegression())
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

new_car = pd.DataFrame([{'Year':2015, 'Present_Price':7.534, 'Driven_kms':36666, 'Fuel_Type':'Diesel','Selling_type':'Individual','Transmission': 'Manual','Owner':0}])
predicted_price = pipeline.predict(new_car)

print("Predicted Price:", predicted_price[0])
