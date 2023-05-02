import numpy as np

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