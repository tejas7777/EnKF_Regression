import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.dnn import DNNModel

class ModelTrainSGD:
    
    def __init__(self, model):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=1e-3)

    def load_data(self, data, target, set_standardize=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.2, random_state=42)

        if set_standardize:
            self.standardize_data()

        self.__convert_data_to_tensor()

    def standardize_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def __convert_data_to_tensor(self):
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

    def train(self, num_epochs=10):
        print("TRAINING STARTED ...")
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(self.X_train)
            loss = self.loss_function(output, self.y_train)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(self.X_test)
            test_loss = self.loss_function(test_output, self.y_test)
        print(f'Test Loss: {test_loss.item()}')

# Just a Test Regression Dataset
diabetes_dataset = load_diabetes()

# Example usage:
model_train_sgd = ModelTrainSGD(model=DNNModel())
model_train_sgd.load_data(data=diabetes_dataset.data, target=diabetes_dataset.target)
model_train_sgd.train(num_epochs=100)
model_train_sgd.evaluate()