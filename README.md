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
This part implements a reinforcement learning (RL) approach for reconfiguring shipboard power systems. It utilizes a graph-based representation of the shipboard power system and employs the Proximal Policy Optimization (PPO) algorithm to optimize the system's configuration.
Key Features:
Simulink Integration: Uses MATLAB engine to interface with a Simulink model of a two-zone Medium Voltage DC (MVDC) shipboard power system.
Custom Gym Environment: Implements a custom OpenAI Gym environment (ShipEnvironment) to simulate the shipboard power system.
PPO Agent: Implements a PPO agent with separate actor and critic networks for learning optimal reconfiguration strategies.
Dynamic Action Space: Utilizes a binary action space to control various switches and components in the power system.
Reward Function: Incorporates a reward function based on generator power outputs, bus voltages, and load powers.
Visualization: Includes plotting functionality to track training progress, including rewards and losses.
Files
pyinterface.py: Handles the interface between Python and MATLAB/Simulink, including the custom Gym environment.
TrainingPPO.py: Implements the PPO algorithm and training loop for the reconfiguration task.
Usage
This project is designed for researchers and engineers working on intelligent control and optimization of shipboard power systems. It demonstrates how reinforcement learning can be applied to complex, graph-based power system reconfiguration problems.
Requirements
Python 3.x
TensorFlow
OpenAI Gym
MATLAB Engine for Python
NumPy
Matplotlib
