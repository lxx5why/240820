import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


data = [[1,2,3], [4,5,6]]
df = pd.DataFrame(data)

data = pd.read_csv("data/5.HeightWeight.csv", index_col=0)

X = data['Height(Inches)']*2.54
Y = data['Weight(Pounds)']*0.453592


array = data.values

X = array[:, 1]
print(X)
Y = array[:, 0]
print(Y)
# X = X.values
X = X.reshape(-1, 1)

# 데이터 분할
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)
# model.coef_
# model.intercept_

Y_pred = model.predict(X_test)
print(Y_pred)
df_Y_test = pd.DataFrame(Y_test)


plt.figure(figsize=(10,6))
plt.scatter(X_test[:100], Y_test[:100], color='blue', label='Actual Values')
plt.scatter(X_test[:100], Y_pred[:100], color='red', label='Predict Values', marker='x')

plt.title("Index")
plt.xlabel("Height(cm)")
plt.ylabel("Weight(kg)")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')

MAE = mean_absolute_error(Y_test, Y_pred)
print(MAE)