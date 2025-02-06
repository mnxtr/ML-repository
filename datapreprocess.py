import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load dataset (Replace with actual EKG dataset)
data = pd.read_csv("ekg_data.csv")
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train base models
rf = RandomForestClassifier(n_estimators=100)
gb = GradientBoostingClassifier(n_estimators=100)
svm = SVC(probability=True, kernel='rbf')
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

models = [rf, gb, svm, xgb]
for model in models:
    model.fit(X_train, y_train)

# Meta-learning using Logistic Regression
from sklearn.linear_model import LogisticRegression
meta_features_train = np.column_stack([model.predict_proba(X_train)[:, 1] for model in models])
meta_features_test = np.column_stack([model.predict_proba(X_test)[:, 1] for model in models])

meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)

# Predict and Evaluate
y_pred = meta_model.predict(meta_features_test)
print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# CNN-LSTM for EKG signal classification
cnn_lstm_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=True),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(len(set(y)), activation='softmax')  # Multiclass classification
])

cnn_lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_lstm_model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, validation_data=(X_test.reshape(-1, X_test.shape[1], 1), y_test))
