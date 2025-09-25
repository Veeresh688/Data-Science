from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import kagglehub
import os
import pandas as pd

dataset_dir = kagglehub.dataset_download("saurabh00007/iriscsv")
files = os.listdir(dataset_dir)
print("Files in dataset folder:", files)

csv_file = os.path.join(dataset_dir, files[0])
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

df = pd.read_csv(csv_file)
X  = df.drop(['Id','Species'], axis = 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Species'])

x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =42)

model = LogisticRegression(max_iter=500)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print('score: ', score)

new_data = [[5.1, 3.5, 1.4, 0.2]]  
prediction = model.predict(new_data)
species = label_encoder.inverse_transform(prediction)
print(species[0])
