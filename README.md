# Non Intrusive Load Monitoring in Shipboard Power Systems
This project implements a NILM model using a Convolutional Neural Network (CNN) to process time-series data from simulated electrical systems. The dataset is preprocessed with the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to handle class imbalance effectively.


Data Handling:

Loads time-series data from MATLAB .mat files representing simulated electrical system outputs.
Combines multi-source datasets into a structured 3D tensor for CNN processing.
Class Imbalance Handling:

Utilizes the SMOTE algorithm to generate balanced datasets by oversampling minority classes.

Model Architecture:

A 1D Convolutional Neural Network (CNN) with:
Conv1D for feature extraction.
MaxPooling for dimensionality reduction.
Dense layers for classification.
A final softmax activation for multi-class output.
Training and Evaluation:

Splits the dataset into training and testing subsets.
Trains the model using the Adam optimizer and categorical crossentropy loss.
Evaluates model accuracy and generates confusion matrices for detailed performance analysis.
Visualization:

Visualizes data distribution across classes before and after oversampling.
Generates heatmaps for confusion matrices to provide insights into classification performance.


# Non Intrusive Fault Detection in Shipboard Power Systems

demonstrates the implementation of a Graph Neural Network (GNN) for fault classification using the Spektral library. It processes tabular data, constructs a graph structure, and applies graph convolutional layers to classify the data effectively.

Key Features:
Data Preprocessing:

Reads tabular data from a CSV file.
Normalizes input features using StandardScaler.
Handles class imbalance with RandomOverSampler.
Graph Construction:

Constructs an adjacency matrix .
Encodes the dataset into a graph format suitable for GNN processing using Spektral's Graph and Dataset classes.
Graph Neural Network Architecture:

Built using Spektral's GCNConv layers with the following structure:
Two graph convolution layers with ReLU activation.
Dropout layers to prevent overfitting.
A fully connected dense layer with softmax activation for multi-class classification.
Model Training and Evaluation:

Compiles the GNN model with the Adam optimizer and categorical crossentropy loss.
Trains the model with a validation split and plots the training/validation accuracy and loss curves.
Evaluates the model on a test set and generates a confusion matrix to visualize performance.
Visualization:

Heatmap for the confusion matrix to analyze classification results.
Plots training and validation metrics over epochs.

# Network Reconfiguration in Shipboard Power Systems

