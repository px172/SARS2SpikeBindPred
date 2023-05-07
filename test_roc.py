import argparse
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tools import protein_to_onehot

# Define the command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name','-n', default=None, type=str, required=True, help='name of antibody/receptor')
args = parser.parse_args()
abname = args.name

model_filename = os.path.join("model",abname+"_RNN.h5")
test_filename = os.path.join("data",abname+"_test_data.csv")
onehot_filename = os.path.join("data", abname+"_test_onehot.npy")
figure_filename = os.path.join('figures',abname+'_test_roc_curve.png')

# Load the saved model
model = keras.models.load_model(model_filename)

# Print the input shape of the model
print("input model shape: "+str(model.layers[0].input_shape))

# Load the new data from a CSV file
new_data = pd.read_csv(test_filename)

# Convert the protein sequences to one-hot encoding vectors for the new data
onehot = np.array([protein_to_onehot(seq) for seq in new_data['junction_aa']])

# Get the true labels for the new data
#true_labels = new_data['Label']
true_labels = np.array(new_data['Label'].astype('category').cat.codes)

# Make predictions on the new data using the loaded model
predictions = model.predict(onehot)

# Compute the false positive rate, true positive rate, and threshold values for the ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

accuracy = accuracy_score(true_labels, predictions.round())
precision = precision_score(true_labels, predictions.round())
recall = recall_score(true_labels, predictions.round())
f1 = f1_score(true_labels, predictions.round())
roc_auc = roc_auc_score(true_labels, predictions)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 score: {:.2f}".format(f1))
print("ROC AUC score: {:.2f}".format(roc_auc))

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line representing random classification
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the plot as a PNG file
plt.savefig(figure_filename)
print("write ROC figure to "+figure_filename)