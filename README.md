# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

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

## PROGRAM
## Name: Janani.V.S
## Register Number: 212222230050
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

worksheet = gc.open('dlexp1').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'X':'float'})
df = df.astype({'Y':'float'})
df.head(10)

X = df[['X']].values
Y = df[['Y']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train = Scaler.transform(X_train)

AI_brain = Sequential([
    Dense(units = 4, activation = 'relu',input_shape = [1]),
    Dense(units = 3, activation = 'relu'),
    Dense(units = 1)
    ])

AI_brain.summary()

AI_brain.compile(optimizer = 'rmsprop', loss = 'mse')

AI_brain.fit(X_train, Y_train, epochs = 9000)

loss_df = pd.DataFrame(AI_brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

X_new = [[9]]

X_neww = Scaler.transform(X_new)

AI_brain.predict(X_neww)
```

## Dataset Information

![image](https://github.com/janani225/basic-nn-model/assets/113497333/9a3284a1-07d6-4880-a4ca-a88914d1a7bf)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/janani225/basic-nn-model/assets/113497333/d72d8ea5-cffb-490e-9905-d25d20f18344)


### Test Data Root Mean Squared Error

![image](https://github.com/janani225/basic-nn-model/assets/113497333/56cbf879-1a22-4fab-929e-741255379dc8)


### New Sample Data Prediction

![image](https://github.com/janani225/basic-nn-model/assets/113497333/2746d6da-b918-4c4f-8d23-c995ab7020f3)



## RESULT
A neural network regression model for the given dataset has been developed Sucessfully.
