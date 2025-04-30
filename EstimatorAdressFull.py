import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read parquet file
house_pdf = pd.read_parquet("dataset/DKHousingPrices.parquet")

# Define feature sets
numerical_features = [
    'sqm',
    'dk_ann_infl_rate%',  # Inflation rate (use as numerical input)
    'year_build',
]

categorical_features = [
    'zip_code',
    'house_type'
]

drop_features = [
    'house_id',
    'sales_type',
]

# drop features
house_pdf = house_pdf.drop(drop_features, axis=1)

# Add a feature for the age of the house: "years_since_build"
house_pdf['years_since_build'] = 2025 - house_pdf['year_build']
numerical_features.append('years_since_build')  # Add this to the list of numerical features

# One-hot encode categorical features
house_pdf_encoded = pd.get_dummies(house_pdf, columns=categorical_features)

print("Encoded DataFrame:")
print(house_pdf_encoded.head())

house_pdf_encoded = house_pdf_encoded.dropna()

# Separate the target variable
y = torch.tensor(house_pdf_encoded['purchase_price'].values, dtype=torch.float).view(-1, 1)

# Select numerical features for scaling
X_numerical = house_pdf_encoded[numerical_features]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

# Convert the scaled features and target to torch tensors
X = torch.tensor(X_scaled, dtype=torch.float)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=2)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the neural network model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Define the loss function and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Track loss
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

# Save the model
torch.save(model, 'models/house_price_model.pth')
print("Model saved to models/house_price_model.pth")

# Estimating the price on new data
new_house = {
    'sqm': 165,
    'zip_code': 8000,
    'dk_ann_infl_rate%': 2.3,
    'house_type': 'Villa',
    'year_build': 2020,
}

# Add the same "years_since_build" feature to the new house data
new_house['years_since_build'] = 2024 - new_house['year_build']

# One-hot encode the categorical values for new data
new_house_df = pd.DataFrame([new_house])

# Apply the same encoding as during training (excluding 'purchase_price')
new_house_df_encoded = pd.get_dummies(new_house_df)

# Align columns with the training data (if new data has missing columns)
new_house_df_encoded = new_house_df_encoded.reindex(columns=house_pdf_encoded.columns, fill_value=0)

# Prepare the new data for prediction
new_house_scaled = scaler.transform(new_house_df_encoded[numerical_features])
X_new = torch.tensor(new_house_scaled, dtype=torch.float)

# Predict the price
model.eval()
with torch.no_grad():
    predicted_price = model(X_new).item()

print(f"Estimated price: {predicted_price:,.2f} DKK")
