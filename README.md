# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1130" height="787" alt="545694014-a808b063-19e9-4071-beed-5a1c23a8988b" src="https://github.com/user-attachments/assets/5e891608-5ace-4527-be60-cb20cbe5971a" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

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

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:J.JANANI

### Register Number:212223230085

```python

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/dl exp1.csv')
X = dataset1['Input'].values
y = dataset1['Output'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Name:J.JANANI
# Register Number:212223230085
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)

# Name:SUBHASHRI R
# Register Number:212223230219
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)


with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')


loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
     

```

### Dataset Information

<img width="507" height="374" alt="545696064-c36654a0-0506-4c23-86e9-e66bef621696" src="https://github.com/user-attachments/assets/70c6cdb6-b6e4-4110-a8f4-fb69941f77f9" />

### OUTPUT
<img width="370" height="233" alt="545696071-7bdb328e-1765-469f-b958-85ad1953d710" src="https://github.com/user-attachments/assets/cc1ef9dc-e293-402f-bc85-9f3da951ef3b" />

### Training Loss Vs Iteration Plot
<img width="677" height="461" alt="545697703-294c1f2c-3d28-4bf3-85f7-a553f969f5d0" src="https://github.com/user-attachments/assets/46c5d6d7-aa9e-41b5-ba40-a5bead9c0d1b" />


### New Sample Data Prediction

<img width="247" height="27" alt="545697539-7eff109a-7af2-4e3b-91f5-56543fa46604" src="https://github.com/user-attachments/assets/50f4c062-0f81-4458-a45c-a82638a4ef84" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
