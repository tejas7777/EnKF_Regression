import torch
import copy

class EnKFOptimizer:
    def __init__(self, model, lr=1e-3, sigma=0.1, k=100, gamma=1e-3, max_iterations=100):
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.parameters = list(model.parameters())
        self.theta = torch.cat([p.data.view(-1) for p in self.parameters])  # Flattened parameters
        self.shapes = [p.shape for p in self.parameters]  # To keep track of original shapes
        self.cumulative_sizes = [0] + list(torch.cumsum(torch.tensor([p.numel() for p in self.parameters]), dim=0))

    def flatten_parameters(self, parameters):
        return torch.cat([p.data.view(-1) for p in parameters])

    def unflatten_parameters(self, flat_params):
        params_list = []
        start = 0
        for shape in self.shapes:
            num_elements = torch.prod(torch.tensor(shape))
            params_list.append(flat_params[start:start + num_elements].view(shape))
            start += num_elements
        return params_list
    


    def step(self, F, D):
        for _ in range(self.max_iterations):
            '''
            Step [1] Draw K Particles
            '''
            self.Omega = torch.randn((self.theta.numel(), self.k)) * self.sigma  # Draw particles
            particles = self.theta.unsqueeze(1) + self.Omega  # Add noise to the current parameter estimate
            # Convert particles back to the parameter shapes for each model evaluation
            perturbed_parameters_list = [self.unflatten_parameters(particles[:, i]) for i in range(self.k)]

            '''
            Step [2] Evaluate the forward model
            '''
            current_params_unflattened = self.unflatten_parameters(self.theta)
            F_current = F(current_params_unflattened)  # Assuming F returns the raw output
            Q = torch.zeros(1, self.k)  # Initialize Q to store the differences

            #print(f"F CURRENT {F_current.shape}")

            for i in range(self.k):
                # Create perturbed parameters for particle i
                perturbed_params = particles[:, i]
                perturbed_params_unflattened = self.unflatten_parameters(perturbed_params)

                # Evaluate the forward model on the perturbed parameters
                F_perturbed = F(perturbed_params_unflattened)

                # Compute the difference and store it in Q
                Q[0, i] = (F_perturbed - F_current).mean().item()  # Store mean difference for scalar output

            #print(f"Shape of Q: {Q.shape}")  # Should be [1, k]

            '''
            Step [3] r Hj = Qj(transpose) x Qj + Γ
            '''
            H_j = Q.T @ Q + self.gamma * torch.eye(self.k)
            H_inv = torch.inverse(H_j)

            '''
            Step [4] Calculate the Gradient of loss function with respect to the current parameters
            '''
            gradient = self.calculate_gradient(F, D)
            gradient = gradient.view(-1, 1)  # Ensure it's a column vector
            
            '''
            Step [5] Update the paramters
            '''
            # print(f"Shape of Omega: {self.Omega.shape}")  # Should be [n, k]
            # print(f"Shape of H_inv: {H_inv.shape}")      # Should be [k, k]
            # print(f"Shape of Q Transpose (Q.T): {Q.T.shape}")  # Should be [k, 1]
            # print(f"Shape of Gradient: {gradient.shape}")      # Should be [n, 1]

            adjustment = H_inv @ Q.T  # This results in [k, 1]
            scaled_adjustment = self.Omega @ adjustment  # Proper multiplication, results in [n, 1]
            update = scaled_adjustment * gradient  # Element-wise multiplication to scale the update by the gradient

            # print("Update calculated successfully")
            # print(f"Shape of Update: {update.shape}")  # Should be [n, 1]

            # Reshape update to be a flat tensor before applying it to theta
            update = update.view(-1)  # Reshape to [n]

            self.theta -= self.lr * update  # Now both are [n], so the operation is valid

            # Update the actual model parameters
            self.update_model_parameters(self.theta)


    def update_model_parameters(self, flat_params):
        idx = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            param.data.copy_(flat_params[idx:idx + num_elements].reshape(param.shape))
            idx += num_elements



    def calculate_gradient(self, F, loss, epsilon=1e-5):
        grad = torch.zeros_like(self.theta)  # Gradient tensor initialized to zero

        for i in range(len(self.theta)):
            # Save original parameter value
            original_value = self.theta[i].item()

            # Perturb parameter positively
            self.theta[i] = original_value + epsilon
            loss_plus = loss(F(self.unflatten_parameters(self.theta)))

            # Perturb parameter negatively
            self.theta[i] = original_value - epsilon
            loss_minus = loss(F(self.unflatten_parameters(self.theta)))

            # Approximate derivative (finite difference)
            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore original parameter value
            self.theta[i] = original_value

        return grad







