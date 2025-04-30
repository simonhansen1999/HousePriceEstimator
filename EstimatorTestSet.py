import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import kagglehub

#path = kagglehub.dataset_download("martinfrederiksen/danish-residential-housing-prices-1992-2024")

#print("Path to dataset files:", path)

house_pdf = pd.read_csv('dataset/DKHousingPrices.csv')

print(house_pdf.head())

numerical_features = [
    'year_build',
    '%_change_between_offer_and_purchase',
    'no_rooms',
    'sqm',
    'sqm_price',
    'nom_interest_rate%',
]

categorical_features = [
    'date',
    'quarter',
    'house_type',
    'sales_type',
    'address',
    'zip_code',
    'city',
    'area',
    'region'
]

drop_features = [
    'house_id',
    'sales_type',
]

# drop features
house_pdf = house_pdf.drop(drop_features, axis=1)

# create Tensors
X = torch.tensor(house_pdf[numerical_features].values, dtype=torch.float)
y = torch.tensor(house_pdf['purchase_price'].values, dtype=torch.float).view(-1, 1)

# normalize the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=2)

# set random seed for reproducibility
torch.manual_seed(42)

# define the model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# define the loss function and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# define the training loop
num_epochs = 1000

for epoch in range(num_epochs):
    predictions = model(X_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

torch.save(model, 'models/house_price_model.pth')

print("Model saved to models/house_price_model.pth")

# Estimate price on new data
new_house = {
    'year_build': 2015,
    '%_change_between_offer_and_purchase': -2.5,
    'no_rooms': 4,
    'sqm': 120,
    'sqm_price': 21000,
    'nom_interest_rate%': 3.5,
}

# Step 2: Convert to DataFrame and tensor
new_df = pd.DataFrame([new_house])
X_new = torch.tensor(new_df.values, dtype=torch.float)

# Step 3: Predict price
model.eval()

with torch.no_grad():
    predicted_price = model(X_new).item()

print(f"Estimated price: {predicted_price:,.2f} DKK")

#loaded_model = torch.load('model/house_price_model.pth')

#loaded_model.eval()

#with torch.no_grad():
#    predictions = loaded_model(X_test)
#    test_MSE = loss(predictions, y_test)