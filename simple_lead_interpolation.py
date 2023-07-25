"""
Simple Lead Interpolation With CNN Deep Learning Network

Jacob Casey
Date: March 26, 2023
"""

import pandas as pd
import numpy as np
import os
import wfdb
import ast
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Function to load raw data
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# Function to preprocess data
def preprocess_data(X):
    # Filter the signal using a bandpass filter
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * sampling_rate
    low = 0.1 / nyquist
    high = 15 / nyquist
    order = 4
    b, a = butter(order, [low, high], btype='band')
    X_filtered = filtfilt(b, a, X, axis=1)
    
    # Normalize the signal to have zero mean and unit variance
    X_normalized = (X_filtered - np.mean(X_filtered, axis=1, keepdims=True)) / np.std(X_filtered, axis=1, keepdims=True)
    
    # Resample the signal to 250 Hz
    from scipy.signal import resample
    X_resampled = resample(X_normalized, int(X_normalized.shape[1] * 250 / sampling_rate), axis=1)
    
    return X_resampled.reshape(-1, X_resampled.shape[1], 1)

# Function to encode the target variable
def encode_labels(y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder.transform(y)

# Function to create the deep learning model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

"""
Step 1: Load the data

Student: Jacob Casey
Date: March 26, 2023
"""
data_directory = 'C:/Users/Jacob/University/KCL/Research Proj/data/ptb_xl_v1.0.3/records100/'

# Function to load a single ECG record
def load_ecg_record(file_path):
    signal, metadata = wfdb.rdsamp(file_path)
    return signal, metadata

# Function to traverse the dataset folder structure and load all records
def load_all_records(data_directory):
    all_records = []
    record_count = 0
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.hea'):
                record_path = os.path.join(root, file[:-4])  # Removing '.hea' extension
                signal, metadata = load_ecg_record(record_path)
                all_records.append((signal, metadata))
                record_count += 1
                print(f'Processed {record_count} records: {record_path}')
    return all_records

# Load all the ECG records
all_records = load_all_records(data_directory)

# Print the number of records
print(f"Number of records: {len(all_records)}")

# Print the signal data shape and metadata for the first record
print('Signal data shape:', all_records[0][0].shape)
print('Metadata:', all_records[0][1])

# Get the names of the leads for the first record
lead_names = all_records[0][1]['sig_name']
print('Lead names:', lead_names)


# Preprocess data
X_processed = preprocess_data(X)

# Encode target variable
y_encoded = encode_labels(y)

# Split data into training and testing sets
test_fold = 10
X_train = X_processed[df['strat_fold'] != test_fold]
X_test = X_processed[df['strat_fold'] == test_fold]
y_train = y_encoded[df['strat_fold'] != test_fold]
y_test = y_encoded[df['strat_fold'] == test_fold]

# Create and train the deep learning model
batch_size = 32
epochs = 10
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
score, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', accuracy)

# Perform lead interpolation on a new ECG signal
new_signal = np.array([[-0.112, 0.224, 0.336, 0.224, -0.112], [-0.224, 0.448, 0.672, 0.448, -0.224]])
new_signal_processed = preprocess_data(new_signal)
predicted_labels = model.predict(new_signal_processed)
predicted_labels_decoded = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
print('Predicted labels:', predicted_labels_decoded)