import argparse
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tools import protein_to_onehot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from numpy import nan_to_num
import tensorflow as tf

# editted
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        y_pred_binary = (y_pred > 0.5).astype(int)
        conf_mat = confusion_matrix(self.y_val, y_pred_binary)
        TP = conf_mat[1,1]
        TN = conf_mat[0,0]
        FP = conf_mat[0,1]
        FN = conf_mat[1,0]
        mcc = matthews_corrcoef(self.y_val, y_pred_binary)
        mcc = nan_to_num(mcc, nan=0)
        print(f'Confusion matrix at epoch {epoch}:')
        conf_df = pd.DataFrame(conf_mat, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        print(conf_df)
        print("MCC = {:.3f}".format(mcc))

# Define the command-line argument parser
parser = argparse.ArgumentParser()

# Example
parser.add_argument('--name','-n', default=None, type=str, required=True,  help='name of antibody/receptor')
parser.add_argument('--force','-f', action="store_true", default=None, required=False,  help='fore to retrain the modle')
parser.add_argument('--epochs', '-e', default=1, required=False, type=int, help='Number of epochs')
#parser.add_argument('abname', help='the name of the antibody/receptor')

# Parse the command-line arguments
args = parser.parse_args()
abname = args.name

#
if args.epochs != 1:
    num_epochs =  args.epochs

# Define the filename of the CSV file
filename = os.path.join("data",abname+"_train_data.csv")
model_filename = os.path.join("modelV2",abname+"_RNN_epochs"+str(num_epochs)+".h5")
#onehot_filename = os.path.join("data", abname+"_train_onehot.npy")  # new line
model_fig_filename = os.path.join("figures", abname+"_RNN_model.png")
accuracy_fig_filename = os.path.join("figures", abname+"_trainV2"+"_RNN_epochs"+str(num_epochs)+"_accuracy.png")
loss_fig_filename = os.path.join("figures", abname+"_trainV2"+"_RNN_epochs"+str(num_epochs)+"_loss.png")

if os.path.exists(model_filename):
    if args.force:
        print("The model file exists. Retrain the model with --force .")
    else:
        overwrite = input("Do you want to retrain the model? Press 'y' to confirm: ")
        if overwrite == "y":
            print("The model file will be overwritten.")
        else:
            print("The model will not be retrained. exit.")
            os._exit(0)

# Load the data from the CSV file
df = pd.read_csv(filename)

# Convert the protein sequences to one-hot encoding vectors
onehot = np.array([protein_to_onehot(seq) for seq in df['junction_aa']])

# Convert the labels to integers
labels = np.array(df['Label'].astype('category').cat.codes)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(onehot, labels, test_size=0.2, random_state=42)

# Define the LSTM model
model = keras.Sequential()
model.add(layers.LSTM(units=80, return_sequences=True, dropout=0.2, input_shape=(onehot.shape[1], onehot.shape[2])))
model.add(layers.LSTM(units=80, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(units=80, return_sequences=False, dropout=0.2))
model.add(layers.Dense(units=50, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(curve='PR')])

# Train the model

confusion_matrix_callback = ConfusionMatrixCallback(X_val, y_val)
#history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[confusion_matrix_callback])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=32, callbacks=[confusion_matrix_callback])
#history = model.fit(onehot, labels, validation_split=0.1, verbose=2, epochs=num_epochs, batch_size=32)
model.save(model_filename)

plot_model(model, show_shapes=True, dpi=150, to_file=model_fig_filename)

print("The model is saved at "+model_filename)

# Plot the training and validation accuracy over time
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Model Accuracy of '+abname)
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.legend(['train', 'val'], loc='upper left')
#plt.show()
#plt.savefig(accuracy_fig_filename)
#plt.close

# Plot the training and validation loss over time
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss of '+abname)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(loss_fig_filename)
plt.close
print("The loss figure is saved at "+loss_fig_filename)