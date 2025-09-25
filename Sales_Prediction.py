import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Download latest version
dataset_dir = kagglehub.dataset_download("bumba5341/advertisingcsv")

files = os.listdir(dataset_dir)
print("Files in dataset folder:", files)

csv_file = os.path.join(dataset_dir, files[0])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(csv_file)
X = df[['TV', 'Radio', 'Newspaper']]
Y = df[['Sales']]

scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print('Score: ', r2_score(y_test,y_pred))

new_data = [[230.1, 37.8,69.2]]
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print('Sales prediction: ', prediction[0])
