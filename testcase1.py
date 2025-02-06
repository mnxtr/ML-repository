################################################################################
# 1. IMPORT LIBRARIES
################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# For PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For scikit-learn classifiers and metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

################################################################################
# 2. DATA LOADING
################################################################################

class ECGDataset(Dataset):
    """
    Custom PyTorch Dataset for ECG data. 
    Expects data to be stored in a suitable format, e.g.:
        - A set of segmented ECG beats or windows
        - Corresponding arrhythmia labels
    """
    def __init__(self, data_array, label_array, transform=None):
        super().__init__()
        self.data = data_array          # Shape: (num_samples, signal_length)
        self.labels = label_array       # Numeric or categorical classes
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ecg_signal = self.data[index]
        label = self.labels[index]

        if self.transform:
            ecg_signal = self.transform(ecg_signal)

        # Convert to PyTorch tensors
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return ecg_signal, label

def load_ecg_data(data_path):
    """
    Outline function to load ECG data from local files or a database.
    Returns:
        data (numpy array): shape = (num_samples, signal_length)
        labels (numpy array): shape = (num_samples,)
    """
    # Pseudocode: adapt to your data format
    # -------------------------------------------------
    # Example:
    # df = pd.read_csv(os.path.join(data_path, "ecg_signals.csv"))
    # data = df.iloc[:, :-1].values  # all columns except last
    # labels = df.iloc[:, -1].values  # last column as labels
    #
    # return data, labels
    pass  # Replace with actual data loading logic

################################################################################
# 3. PREPROCESSING
################################################################################

def preprocess_ecg_signals(raw_data):
    """
    Any baseline wander removal, noise filtering, normalization, etc.
    Returns preprocessed data of the same shape.
    """
    # TODO: Implement filtering, normalization
    # For demonstration, we'll just scale signals
    scaled_data = (raw_data - np.mean(raw_data, axis=1, keepdims=True)) / \
                  (np.std(raw_data, axis=1, keepdims=True) + 1e-8)
    return scaled_data

################################################################################
# 4. DEEP LEARNING FEATURE EXTRACTION
################################################################################

class SimpleECGCNN(nn.Module):
    """
    A basic CNN architecture for feature extraction from ECG signals.
    This network can be adjusted for more complex designs (e.g., ResNet-like).
    """
    def __init__(self, input_length, num_features=128):
        super(SimpleECGCNN, self).__init__()

        # Example architecture: 1D Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Compute the dimension after conv/pool layers to flatten
        # Assuming input shape is (batch_size, 1, input_length)
        # So let's approximate final dimension:
        dummy_input = torch.zeros(1, 1, input_length)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.shape[1] * dummy_output.shape[2]

        # Fully connected layer to produce "embedding"
        self.fc = nn.Linear(flattened_size, num_features)

    def _forward_conv(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        """
        Forward pass returning an embedding vector rather than classification.
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)             # Feature vector of shape (batch, num_features)
        return x

def train_feature_extractor(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """
    Train the CNN model to learn discriminative ECG features.
    In this scenario, we might do self-supervised or supervised learning.
    For supervised, we could add a classification head temporarily.
    """
    # Example: adding a simple linear classification head for training
    classification_head = nn.Linear(model.fc.out_features, NUM_CLASSES)

    parameters = list(model.parameters()) + list(classification_head.parameters())
    optimizer = optim.Adam(parameters, lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        classification_head.train()

        for batch_data, batch_labels in train_loader:
            # batch_data shape: (batch_size, signal_length)
            # reshape for 1D conv
            batch_data = batch_data.unsqueeze(1)  # (batch_size, 1, signal_length)

            optimizer.zero_grad()
            features = model(batch_data)
            preds = classification_head(features)
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()

        # Validation step (optional)
        model.eval()
        classification_head.eval()
        # Evaluate val_loader, compute val_loss or val_accuracy, etc.

    # Optionally, remove the classification head after training
    return model

def extract_features(model, data_loader):
    """
    Pass data through the trained CNN (without a classification head)
    and return a matrix of extracted features.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.unsqueeze(1)
            features = model(batch_data)
            all_features.append(features.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels

################################################################################
# 5. CLASSIFICATION WITH TRADITIONAL ML
################################################################################

def train_classical_model(features, labels, model_type='rf'):
    """
    Train a traditional ML classifier (e.g. Random Forest or SVM).
    model_type: 'rf' or 'svm'
    """
    if model_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        classifier = SVC(kernel='rbf', probability=True, random_state=42)
    else:
        raise ValueError("Unknown model type")

    # Optionally scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train the chosen classifier
    classifier.fit(scaled_features, labels)
    return classifier, scaler

def evaluate_classical_model(classifier, scaler, features, labels):
    """
    Evaluate the trained classical model on a test set.
    """
    scaled_features = scaler.transform(features)
    preds = classifier.predict(scaled_features)

    print("Classification Report:")
    print(classification_report(labels, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

################################################################################
# 6. END-TO-END WORKFLOW
################################################################################

def main():
    # 1. Load the data
    # --------------------------------------------------
    data_path = "./ecg_data"
    raw_data, raw_labels = load_ecg_data(data_path)  # shape (num_samples, signal_length), (num_samples,)

    # 2. Preprocess the data
    # --------------------------------------------------
    preprocessed_data = preprocess_ecg_signals(raw_data)

    # 3. Split into train/val/test
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(preprocessed_data, raw_labels,
                                                        test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=0.5, random_state=42)

    # 4. Create Datasets & DataLoaders
    # --------------------------------------------------
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset   = ECGDataset(X_val, y_val)
    test_dataset  = ECGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 5. Define CNN-based Feature Extractor
    # --------------------------------------------------
    input_length = X_train.shape[1]  # signal_length
    model = SimpleECGCNN(input_length=input_length, num_features=128)

    # 6. Train Feature Extractor (Optional Supervised Approach)
    # --------------------------------------------------
    global NUM_CLASSES
    NUM_CLASSES = len(np.unique(y_train))
    trained_model = train_feature_extractor(model, train_loader, val_loader,
                                            epochs=10, lr=1e-3)

    # 7. Extract Features from Training and Test Data
    # --------------------------------------------------
    train_features, train_feat_labels = extract_features(trained_model, train_loader)
    test_features, test_feat_labels = extract_features(trained_model, test_loader)

    # 8. Train Classical ML Classifier on Extracted Features
    # --------------------------------------------------
    classifier, scaler = train_classical_model(train_features, train_feat_labels, model_type='rf')

    # 9. Evaluate on Test Set
    # --------------------------------------------------
    evaluate_classical_model(classifier, scaler, test_features, test_feat_labels)

    # Potentially do additional steps: cross-validation, multiple model comparisons, etc.

if __name__ == "__main__":
    main()
