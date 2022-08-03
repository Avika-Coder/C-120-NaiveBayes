from google.colab import files
data_to_load=files.upload()

import pandas as pd
import csv
import plotly.express as px
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
col_names = ['glucose', 'bloodpressure', 'diabetes']
df = pd.read_csv("diabetes1.csv", names=col_names).iloc[1:]
print(df.head())

from sklearn.model_selection import train_test_split
X=df[['glucose','bloodpressure']]
y=df['diabetes']
x_train_1,x_test_1,y_train_1,y_test_1=train_test_split(X,y,test_size=0.25,random_state=42)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train_1=sc.fit_transform(x_train_1)
x_test_1=sc.fit_transform(x_test_1)
model_1=GaussianNB()
model_1.fit(x_train_1,y_train_1)
prediction=model_1.predict(x_test_1)
accuracy=accuracy_score(y_test_1,prediction)
print(accuracy)
