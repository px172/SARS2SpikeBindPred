import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def protein_to_onehot(protein_sequence):
    """
    Convert a protein sequence to a one-hot encoding vector with 20 types of amino acids plus one "*" type.

    Args:
        protein_sequence (str): The protein sequence to convert.

    Returns:
        A numpy array of shape (len(protein_sequence), 21), where each row is a one-hot encoding vector of a single amino acid.
    """
    # Define a dictionary that maps amino acid codes to integers from 0 to 20
    amino_acids = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, '*': 20}
    # Initialize a numpy array of zeros with shape (len(protein_sequence), 21)
    onehot = np.zeros((len(protein_sequence), 21))
    # Convert each amino acid in the protein sequence to a one-hot encoding vector
    for i, aa in enumerate(protein_sequence):
        if aa in amino_acids:
            index = amino_acids[aa]
            onehot[i, index] = 1
        else:
            # If the amino acid is not in the dictionary, use the "*" type
            onehot[i, 20] = 1
    return onehot

# Define the filenames of the CSV files
train_filename = os.path.join("data","ACE2_train_data.csv")
test_filename = os.path.join("data","ACE2_test_data.csv")
model_filename = os.path.join("model","ACE2_RNN.h5")

# Load the data from the CSV files
train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)

# Convert the protein sequences to one-hot encoding vectors for train and test data
train_onehot = np.array([protein_to_onehot(seq) for seq in train_df['junction_aa']])
test_onehot = np.array([protein_to_onehot(seq) for seq in test_df['junction_aa']])

# Convert the labels to integers for train and test data
train_labels = np.array(train_df['Label'].astype('category').cat.codes)
test_labels = np.array(test_df['Label'].astype('category').cat.codes)

# Define the LSTM model
model = keras.Sequential()
model.add(layers.LSTM(units=80, return_sequences=True, dropout=0.2, input_shape=(train_onehot.shape[1], train_onehot.shape[2])))
model.add(layers.LSTM(units=80, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(units=80, return_sequences=False, dropout=0.2))
model.add(layers.Dense(units=50, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(curve='PR')])

# Train the model using the training data
#history = model.fit(train_onehot, train_labels, validation_split=0.2, epochs=50, batch_size=32)
history_full = model.fit(train_onehot, train_labels, batch_size = 32, epochs = 50, verbose = 2, validation_split = 0.1)

# Evaluate the model using the test data
test_loss, test_accuracy, test_auc = model.evaluate(test_onehot, test_labels)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save the trained model to a file
model.save('RNN_ACE2_model.h5')
