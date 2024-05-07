# Developing a Neural Network Regression Model
## AIM:
To develop a neural network regression model for the given dataset.

## THEORY:
Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type. A neural network regression model uses interconnected layers of artificial neurons to learn the mapping between input features and a continuous target variable. It leverages activation functions like ReLU to capture non-linear relationships beyond simple linear trends. Training involves minimizing the loss function (e.g., Mean Squared Error) through an optimizer (e.g., Gradient Descent). Regularization techniques like L1/L2 and dropout prevent overfitting. This approach offers flexibility and high accuracy for complex regression problems.

## Neural Network Model:
![output1](https://github.com/AbishekAnand15/basic-nn-model/assets/118706942/a1ffef22-bdbc-491a-a328-b2a7b9085515)



## DESIGN STEPS:

### STEP 1:
Loading the dataset
### STEP 2:
Split the dataset into training and testing
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.

## PROGRAM:
### Name: JANANI.V.S
### Reference No: 212222230050
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('exp no 1').sheet1
data=worksheet.get_all_values()
print(data)

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state = 30)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

ai_model=Sequential([
    Dense(units=8,activation='relu',input_shape=[1]),
    Dense(units=9,activation='relu'),
    Dense(units=1)
])

ai_model.compile(optimizer='rmsprop',loss='mse')

ai_model.fit(X_train1,y_train,epochs=20)

loss_df = pd.DataFrame(ai_model.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)
ai_model.evaluate(X_test1,y_test)

X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_model.predict(X_n1_1)
```
## Dataset Information:
![out2](https://github.com/AbishekAnand15/basic-nn-model/assets/118706942/b6411e79-68dd-469f-afb1-a99da9ca727b)

## OUTPUT:
### Training Loss Vs Iteration Plot
![out3](https://github.com/AbishekAnand15/basic-nn-model/assets/118706942/7dd86522-ecc3-452c-87ba-b9aeda7d091b)

### Test Data Root Mean Squared Error
![out4](https://github.com/AbishekAnand15/basic-nn-model/assets/118706942/0921daf2-014a-4ecf-99bd-255403e3a51d)

### New Sample Data Prediction
![out5](https://github.com/AbishekAnand15/basic-nn-model/assets/118706942/f47edad2-0c63-4974-b7b7-74144268ff76)

## RESULT
A neural network regression model for the given dataset has been developed successfully.
