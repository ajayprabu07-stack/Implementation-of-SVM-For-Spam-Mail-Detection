# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Detect File Encoding: Use chardet to determine the dataset's encoding.
Load Data: Read the dataset with pandas.read_csv using the detected encoding.
Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
Train SVM Model: Fit an SVC model on the training data.
Predict Labels: Predict test labels using the trained SVM model.
Evaluate Model: Calculate and display accuracy with metrics.accuracy_score. 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: AJAYPRABU.A
RegisterNumber:  212225220005
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
<img width="871" height="190" alt="image" src="https://github.com/user-attachments/assets/2096d193-4b67-4f4a-84bc-290f0a180d6c" />
<img width="911" height="291" alt="image" src="https://github.com/user-attachments/assets/801d10de-7e75-4517-a59e-5add50e1018a" />
<img width="513" height="330" alt="image" src="https://github.com/user-attachments/assets/67b6c45a-e203-4a1b-bfff-365f8e3abd3e" />
<img width="267" height="205" alt="image" src="https://github.com/user-attachments/assets/7486ac8c-a096-4e33-b1c0-7fb32e386ae6" />
<img width="811" height="222" alt="image" src="https://github.com/user-attachments/assets/24ca0880-22da-48e3-9de9-98c517e4db3c" />
<img width="518" height="154" alt="image" src="https://github.com/user-attachments/assets/c3bbd5b1-0c94-435a-b133-e6e7cd6674e2" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
