# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Gokul C
RegisterNumber:  212223240040
*/

import pandas as pd

data=pd.read_csv("/content/Employee_EX6.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])

data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y=data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

from sklearn.tree import DecisionTreeClassifier,plot_tree

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plot_tree(dt,feature_names=x.columns,class_names=['not left','left'], filled=True)

plt.show()

```

## Output:

![Screenshot 2024-04-02 181147](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/153058321/b8ac6ee3-405f-4423-9801-a7a7f4df6a66)
![Screenshot 2024-04-02 181216](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/153058321/a4ae7885-4e91-4090-a032-4ac5151e8c37)
![Screenshot 2024-04-02 181232](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/153058321/c45a6de3-e12d-47e8-aa87-5ab3bab2b635)
![Screenshot 2024-04-02 181248](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/153058321/96e67612-84f8-4667-96b6-7dc761d1fda7)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
