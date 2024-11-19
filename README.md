# Non Intrusive Load Monitoring in Shipboard Power Systems
This project implements a NILM model using a Convolutional Neural Network (CNN) to process time-series data from simulated electrical systems. The dataset is preprocessed with the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to handle class imbalance effectively.

Key Features:
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

# Network Reconfiguration in Shipboard Power Systems

