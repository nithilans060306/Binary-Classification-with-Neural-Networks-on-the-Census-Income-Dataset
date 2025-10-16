# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

### NAME : Nithilan S
### REG NO : 212223240108
## AIM
To develop a Neural Network model that classifies whether a person’s income exceeds $50K per year using demographic and employment data from the Adult Income Dataset.

## THEORY
A Neural Network (NN) is a computational model inspired by the structure of the human brain. It consists of layers of neurons that process data through weighted connections.
For income prediction, the NN learns patterns from features such as age, education, occupation, marital status, and hours worked per week to determine the probability of a person earning more than $50K annually.
By learning these relationships through backpropagation and gradient descent, the network improves prediction accuracy over time.

## DESIGN STEPS
### STEP 1: 
Load the dataset and handle missing values.
### STEP 2: 
Label encode categorical variables and normalize continuous features.

### STEP 3: 
Split the dataset into training and testing sets.

### STEP 4: 
Convert data into PyTorch tensors for model training.

### STEP 5: 
Define a Feed-Forward Neural Network (Fully Connected NN).

### STEP 6: 
Train the model using Binary Cross Entropy Loss and Adam Optimizer.

### STEP 7:
Evaluate the model on test data and calculate accuracy.

## PROGRAM

### Name: Nithilan S
### Register Number: 212223240108

```python
# Step 1: Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Step 2: Load dataset
df = pd.read_csv("income.csv")
df

# Step 3: Separate categorical, continuous, and label columns
cat_col = [col for col in df.columns if df[col].dtype == "object" and col != "income"]
cont_col = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'label']
label_col = 'income'

# Step 4: Encode label column
label_enc = LabelEncoder()
df[label_col] = label_enc.fit_transform(df[label_col])

# Step 5: Encode categorical columns
cat_encoders = {}
for cat in cat_col:
    le = LabelEncoder()
    df[cat] = le.fit_transform(df[cat])
    cat_encoders[cat] = le

# Step 6: Prepare feature and target arrays
x_cats = df[cat_col].values
x_conts = df[cont_col].values
y = df[label_col].values

# Step 7: Scale continuous features
scaler = StandardScaler()
x_conts = scaler.fit_transform(x_conts)

# Step 8: Split data into train and test sets
X_cats_train, X_cats_test, X_conts_train, X_conts_test, y_train, y_test = train_test_split(
    x_cats, x_conts, y, test_size=0.2, random_state=42
)

# Step 9: Convert to PyTorch tensors
cats_train = torch.tensor(X_cats_train, dtype=torch.int64)
conts_train = torch.tensor(X_conts_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)

cats_test = torch.tensor(X_cats_test, dtype=torch.int64)
conts_test = torch.tensor(X_conts_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Step 10: Create Tensor datasets and dataloaders
train_ds = TensorDataset(cats_train, conts_train, y_train)
test_ds = TensorDataset(cats_test, conts_test, y_test)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

# Step 11: Define the TabularModel class
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz=2, hidden_units=50, dropout=0.4):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(dropout)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum([nf for _, nf in emb_szs])
        self.fc1 = nn.Linear(n_emb + n_cont, hidden_units)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, out_sz)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

# Step 12: Set embedding sizes and initialize model
emb_sizes = [(len(df[col].unique()), min(50, (len(df[col].unique())+1)//2)) for col in cat_col]

model = TabularModel(emb_sizes, n_cont=len(cont_col))
torch.manual_seed(42)

# Step 13: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 300

# Step 14: Train the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for cat, cont, y in train_dl:
        optimizer.zero_grad()
        y_pred = model(cat, cont)
        loss = criterion(y_pred, y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dl):.4f}")

# Step 15: Evaluate model performance
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for cat, cont, y in test_dl:
        y_pred = model(cat, cont)
        loss = criterion(y_pred, y.long())
        test_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == y).sum().item()
print(f"Test Loss: {test_loss/len(test_dl):.4f}, Accuracy: {correct/len(test_ds):.4f}")

# Step 16: Global prediction function
def predict_new(model, input_dict):
    cat_vals = []
    for c in cat_col:
        le = cat_encoders[c]
        val = input_dict[c]
        enc = le.transform([val])[0]
        cat_vals.append(enc)
    cat_tensor = torch.tensor([cat_vals], dtype=torch.int64)
    cont_vals = [input_dict[c] for c in cont_col]
    cont_scaled = scaler.transform([cont_vals])
    cont_tensor = torch.tensor(cont_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(cat_tensor, cont_tensor)
        pred_class = out.argmax(1).item()
        return label_enc.inverse_transform([pred_class])[0]

# Step 17: Example test input
new_person = {
    'age': 28,
    'sex': 'Male',
    'education': 'Assoc-acdm',
    'education-num': 5,
    'marital-status': 'Married',
    'workclass': 'State-gov',
    'occupation': 'Prof-specialty',
    'hours-per-week': 60
}

print("Predicted Income:", predict_new(model, new_person))

```
## OUTPUT

### Accuracy: 

<img width="404" height="96" alt="image" src="https://github.com/user-attachments/assets/8b03efbf-8346-40f2-8603-1d0174e55855" />

### Predictions

<img width="295" height="81" alt="image" src="https://github.com/user-attachments/assets/c5970402-bc74-44ba-9ba1-3354f9ccb20f" />

## RESULT
The TabularModel was successfully trained to predict income categories from categorical and continuous features.
