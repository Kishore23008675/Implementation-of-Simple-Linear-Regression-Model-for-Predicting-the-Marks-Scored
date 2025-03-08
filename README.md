# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KISHORE A
RegisterNumber:  212223110022
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("df.head\n",df.head())
print("df.tail\n",df.tail())
x = df.iloc[:,:-1].values
print("Array value of x:",x)
y = df.iloc[:,1].values
print("Array value of y:",y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Values of y predict:\n",y_pred)
print("Array values of y test:\n",y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## df.head

![Screenshot 2025-03-08 142157](https://github.com/user-attachments/assets/cf63fe30-97c5-4ebc-bf8f-f6bb1cdae51f)

## df.tail

![Screenshot 2025-03-08 142208](https://github.com/user-attachments/assets/b6b5bedd-730d-40e6-97c0-88467adaa0b1)

## Array value of X

![Screenshot 2025-03-08 142245](https://github.com/user-attachments/assets/8301e1b6-f614-41f7-b06a-d2ada3859b6e)

## Array value of Y

![Screenshot 2025-03-08 142312](https://github.com/user-attachments/assets/de4d861c-55f2-487f-9e2f-e7a791acbf7e)

## Values of Y prediction

![Screenshot 2025-03-08 142322](https://github.com/user-attachments/assets/b741b9fc-f761-4313-9a1c-e07658346e05)

## Array values of Y test

![Screenshot 2025-03-08 142349](https://github.com/user-attachments/assets/45945a7f-bab1-4aef-b460-9c64c7516651)

## Training Set Graph

![Screenshot 2025-03-08 142425](https://github.com/user-attachments/assets/dc16a289-a912-4372-85b3-f25fad620d4b)

## Test set graph

![Screenshot 2025-03-08 142439](https://github.com/user-attachments/assets/87e53c29-d834-4c61-b9cc-79d7fc0c9f58)

## Values of MSE, MAE and RMSE
![Screenshot 2025-03-08 142452](https://github.com/user-attachments/assets/c76f9d7e-1173-4886-9aed-33d546292e6b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
