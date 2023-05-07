import argparse
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tools import protein_to_onehot

# Define the command-line argument parser
parser = argparse.ArgumentParser()

# Example
parser.add_argument('--name','-n', default=None, type=str, required=True,  help='name of antibody/receptor')
parser.add_argument('--force','-f', action="store_true", default=None, required=False,  help='fore to retrain the modle')
#parser.add_argument('abname', help='the name of the antibody/receptor')

# Parse the command-line arguments
args = parser.parse_args()
abname = args.name

# Define the filename of the CSV file
filename = os.path.join("data",abname+"_train_data.csv")
model_filename = os.path.join("model",abname+"_RNN.h5")
onehot_filename = os.path.join("data", abname+"_train_onehot.npy")  # new line

if os.path.exists(model_filename):
    if args.force:
        print("Model file exists. Retrain the model.")
    else:
        overwrite = input("Do you want to retrain the model? Press 'y' to confirm: ")
        if overwrite == "y":
            print("Model file overwritten.")
        else:
            print("The model will not be retrained. exit.")
            os._exit(0)

if os.path.exists(onehot_filename):
    # Load the onehot array from file if it already exists
    print("use the pre-compiled numpy array: "+onehot_filename)
    onehot = np.load(onehot_filename)
else:
    # Load the data from the CSV file
    df = pd.read_csv(filename)

    # Convert the protein sequences to one-hot encoding vectors
    onehot = np.array([protein_to_onehot(seq) for seq in df['junction_aa']])
    # Convert the labels to integers
    labels = np.array(df['Label'].astype('category').cat.codes)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(onehot, labels, test_size=0.2, random_state=42)

    # Save the onehot array to file
    print("save the pre-compiled numpy array: "+onehot_filename)
    np.save(onehot_filename, onehot)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(onehot, labels, test_size=0.2, random_state=42)

# Define the LSTM model
model = keras.Sequential()
model.add(layers.LSTM(units=80, return_sequences=True, dropout=0.2, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.LSTM(units=80, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(units=80, return_sequences=False, dropout=0.2))
model.add(layers.Dense(units=50, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(curve='PR')])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
model.save(model_filename)
print("The model is saved at "+model_filename)
