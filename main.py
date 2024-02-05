import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv("test.csv")

# Preprocess the data
# Handle missing values directly to avoid FutureWarning
data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].median())

# Encode categorical features using OneHotEncoder
encoder = OneHotEncoder()
categorical_features = data[['Gender', 'Customer Type', 'Type of Travel', 'Class']]
categorical_encoded = encoder.fit_transform(categorical_features).toarray()

# Encode the target variable
target = data[['satisfaction']]
target_encoded = encoder.fit_transform(target).toarray()

# Standardize numerical features
scaler = StandardScaler()
numerical_features = data[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']]
numerical_scaled = scaler.fit_transform(numerical_features)

# Concatenate processed features
X = np.concatenate([categorical_encoded, numerical_scaled], axis=1)
y = target_encoded

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Apply early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, callbacks=[early_stopping], batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
