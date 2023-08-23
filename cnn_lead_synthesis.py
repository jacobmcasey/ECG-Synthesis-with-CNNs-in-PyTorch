# Import modules
import pandas as pd
import numpy as np
import os
import wfdb
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from scipy.signal import butter, filtfilt, resample
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split

# Function to encode the target variable
def encode_labels(y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder, label_encoder.transform(y)

# Create architecture  -- SEE LATEX FOR DESIGN FILE
def create_model(input_shape, num_classes):
    model = nn.Sequential(
        nn.Conv1d(input_shape[0], 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(32, 64, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1)
    )
    return model

# Load all the ECG records
X = load_all_records(data_directory)

# Preprocess data
X_processed = preprocess_data(X)

# Encode target variable
label_encoder, y_encoded = encode_labels(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Transform numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# Create and train the deep learning model
model = create_model(X_train.shape[1:], len(np.unique(y_encoded)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training Loop
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(X_train, 0):
        inputs, labels = data, y_train[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.unsqueeze(0))
        loss = criterion(outputs, labels.unsqueeze(0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch: {epoch + 1}, Loss: {running_loss / len(X_train)}')

print('Finished Training')

# Perform lead interpolation on a new ECG signal
# Your new_signal and preprocess_data(new_signal) go here.
new_signal_processed_torch = torch.from_numpy(new_signal_processed).float()
predicted_labels = model(new_signal_processed_torch.unsqueeze(0))
predicted_labels_decoded = label_encoder.inverse_transform(np.argmax(predicted_labels.detach().numpy(), axis=1))
print('Predicted labels:', predicted_labels_decoded)
