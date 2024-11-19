import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
from spektral.data import Dataset, Graph
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Read the CSV file
data = pd.read_csv('input.csv')

# Step 2: Preprocess the data
X = data.iloc[:, 1:7].values  # Features
y = data.iloc[:, -1].values   # Labels

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Oversample the training data to handle class imbalance
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Step 3: Construct a graph for the GNN

def compute_adjacency(features, n_neighbors=5):
    adjacency = adj_graph(features, n_neighbors, mode='connectivity', include_self=True)
    return adjacency.toarray()

# Compute the adjacency matrix for the training data
A = compute_adjacency(X_train_resampled)

# Convert the adjacency matrix and features into a Graph dataset
class GraphDataset(Dataset):
    def read(self):
        return [Graph(x=X_train_resampled, a=A, y=y_train_resampled)]

dataset = GraphDataset()

# Step 4: Create the Graph Neural Network model
# Define GNN layers
input_features = Input(shape=(X_train_resampled.shape[1],))
input_adjacency = Input(shape=(X_train_resampled.shape[0],))

x = GCNConv(32, activation='relu')([input_features, input_adjacency])
x = Dropout(0.5)(x)
x = GCNConv(64, activation='relu')([x, input_adjacency])
x = Dropout(0.5)(x)
output = Dense(len(np.unique(y_train)), activation='softmax')(x)

model = Model(inputs=[input_features, input_adjacency], outputs=output)

# Step 5: Compile and train the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_train_resampled, A],
    y_train_resampled,
    batch_size=32,
    epochs=500,
    validation_split=0.2,
)

# Step 6: Evaluate the model and generate the confusion matrix
y_pred = np.argmax(model.predict([X_test, compute_adjacency(X_test, n_neighbors=5)]), axis=1)
accuracy = np.mean(y_pred == y_test)
confusion_mat = confusion_matrix(y_test, y_pred)

# Step 7: Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Plot accuracy and loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.title('Training and Validation Metrics')
plt.legend()
plt.show()

# Step 9: Print accuracy
print(f'Test Accuracy: {accuracy:.4f}')
