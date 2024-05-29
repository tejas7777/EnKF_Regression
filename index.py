import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optimiser.enkf import EnKFOptimizer  # Import the EnKF optimizer
from model.dnn import DNNModel

class ModelTrain():
    
    def __init__(self,model):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimiser = EnKFOptimizer(model, lr=1e-3, sigma=0.1, k=10, gamma=1e-3, max_iterations=100)

    def load_data(self, data, target, set_standardize = False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.2, random_state=42)

        if set_standardize:
            self.standardizse_data()

        self.__convert_data_to_tensor()

    def standardizse_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def __convert_data_to_tensor(self):
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

    def F(self, parameters):
        with torch.no_grad():  # Ensure no gradients are computed
            # Update the model parameters directly from the list of unflattened tensors
            for original_param, new_param in zip(self.model.parameters(), parameters):
                original_param.data.copy_(new_param.data)

            # Perform the forward pass with the adjusted parameters
            output =  self.model(self.X_train)  # Compute the output of the model

            return output.mean()


    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.optimiser.step(F=self.F, D=self.loss_wrapper)

            self.model.train()
            with torch.no_grad():
                output = self.model(self.X_train)
                loss = self.loss_function(output, self.y_train)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(self.X_test)
            test_loss = self.loss_function(test_output, self.y_test)
        print(f'Test Loss: {test_loss.item()}')

    def loss_wrapper(self, model_output):
        # Assuming model_output is the output from the model
        return self.loss_function(model_output, self.y_train)
    

#Just a Test Rergession Dataset
diabetes_dataset = load_diabetes()

model_train = ModelTrain(model=DNNModel())
model_train.load_data(data=diabetes_dataset.data, target=diabetes_dataset.target)
model_train.train()