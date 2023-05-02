import numpy as np
import pandas as pd
import os
from tensorflow import keras
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tools import protein_to_onehot

model_filename = os.path.join("model","ACE2_RNN.h5")
test_filename = os.path.join("data","ACE2_test_data.csv")

# Load the saved model
model = keras.models.load_model(model_filename)

# Load the new data from a CSV file
new_data = pd.read_csv(test_filename)

# Convert the protein sequences to one-hot encoding vectors for the new data
onehot = np.array([protein_to_onehot(seq) for seq in new_data['junction_aa']])

# Get the true labels for the new data
true_labels = new_data['Label']

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
plt.savefig('roc_curve.png')
