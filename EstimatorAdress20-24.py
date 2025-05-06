import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)  # Use input_size dynamically
        self.layer2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.out(x)
        return x

# Train function
def train():
    house_pdf = pd.read_parquet("dataset/DKHousingPrices.parquet")

    house_pdf['date'] = pd.to_datetime(house_pdf['date'])

    house_pdf['year'] = house_pdf['date'].dt.year

    house_pdf = house_pdf[(house_pdf['year_build'] >= 2020) & (house_pdf['year_build'] <= 2024)]

    house_pdf = onehot_encode(house_pdf, 'zip_code', 'zip')   

    drop_features = [
        'address', '%_change_between_offer_and_purchase', 'house_id', 'dk_ann_infl_rate%', 'yield_on_mortgage_credit_bonds%', 'date', 'sales_type', 'house_type',
        'city', 'area', 'region', 'quarter', 'sqm_price', 'year_build', 'year'
    ]

    house_pdf = house_pdf.drop(drop_features, axis=1)

    house_pdf = house_pdf.dropna()

    X = house_pdf.drop('purchase_price', axis=1).copy()
    y = house_pdf['purchase_price'].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))

    # Save scalers and columns
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(target_scaler, 'models/target_scaler.pkl')
    joblib.dump(list(X.columns), 'models/columns.pkl')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=0.7, random_state=2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model, loss, optimizer
    input_size = X_train.shape[1]  # Dynamically get the input size
    net = Net(input_size)  # Pass the input size to the model
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Batch training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = net(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate model
    net.eval()
    with torch.no_grad():
        predictions = net(X_test)
        mse = loss_fn(predictions, y_test).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()

    print(f"\nAverage test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.2f}")
    print("Training complete.")

    # Save model
    torch.save(net, 'models/house_price_model.pth')
    print("Model saved.")

# Estimate function
def estimate_price():
    model = torch.load('models/house_price_model.pth', weights_only=False)
    scaler = joblib.load('models/scaler.pkl')
    target_scaler = joblib.load('models/target_scaler.pkl')
    columns = joblib.load('models/columns.pkl')

    print("Enter details about the house:")
    sqm = float(input("Square meters (sqm): "))
    no_rooms = float(input("Number of rooms: "))
    zip_code = input("Zip code: ")

    # Create input dict with zeros
    input_dict = {col: 0.0 for col in columns}
    input_dict['sqm'] = sqm
    input_dict['no_rooms'] = no_rooms

    zip_col = f"zip_{zip_code}"
    if zip_col in input_dict:
        input_dict[zip_col] = 1.0
    else:
        print(f"Warning: Zip code {zip_code} was not in training data. Estimation may be unreliable.")

    # Align columns before scaling
    X_input = pd.DataFrame([input_dict])
    X_input = X_input[columns]  # Ensure columns are aligned with training data
    X_scaled = scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_scaled = model(X_tensor).item()
        predicted_price = target_scaler.inverse_transform([[predicted_scaled]])[0][0]

    # Print prediction
    print(f"Estimated price: {predicted_price:,.2f} DKK")

# CLI loop
def main():
    while True:
        choice = input("Commands: 'train', 'est' or 'exit'): ").strip().lower()
        if choice == "train":
            train()
        elif choice == "est":
            estimate_price()
        elif choice == "exit":
            print("Exiting the program.")
            break
        else:
            print("Unknown command. Use 'train', 'est' or 'exit'.")

if __name__ == "__main__":
    main()
