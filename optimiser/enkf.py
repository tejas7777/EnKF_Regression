import torch
import copy

class EnKFOptimizer:
    def __init__(self, model, lr=1e-3, sigma=0.1, k=50, gamma=1e-3, max_iterations=10):
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
            self.Omega = torch.randn((self.theta.numel(), self.k)) * self.sigma  #Draw particles
            particles = self.theta.unsqueeze(1) + self.Omega  #Add noise to the current parameter estimate
            #Convert particles back to the parameter shapes
            perturbed_parameters_list = [self.unflatten_parameters(particles[:, i]) for i in range(self.k)]

            '''
            Step [2] Evaluate the forward model
            '''
            current_params_unflattened = self.unflatten_parameters(self.theta)
            F_current = F(current_params_unflattened)
            Q = torch.zeros(1, self.k)

            for i in range(self.k):
                perturbed_params = particles[:, i]
                perturbed_params_unflattened = self.unflatten_parameters(perturbed_params)

                #Evaluate the forward model on the perturbed parameters
                F_perturbed = F(perturbed_params_unflattened)

                #Compute the difference
                Q[0, i] = (F_perturbed - F_current).mean().item()  #Store mean difference for scalar output


            '''
            Step [3] r Hj = Qj(transpose) x Qj + Γ
            '''
            H_j = Q.T @ Q + self.gamma * torch.eye(self.k)
            H_inv = torch.inverse(H_j)

            '''
            Step [4] Calculate the Gradient of loss function with respect to the current parameters
            '''
            gradient = self.calculate_gradient(F, D)
            gradient = gradient.view(-1, 1)  #Ensure it's a column vector
            
            '''
            Step [5] Update the paramters
            '''

            adjustment = H_inv @ Q.T  #Shape [k, m]
            scaled_adjustment = self.Omega @ adjustment  # Shape [n, m]
            update = scaled_adjustment * gradient
            update = update.view(-1)  # Reshape to [n]

            self.theta -= self.lr * update  #Now both are [n]

            #Update the actual model parameters
            self.update_model_parameters(self.theta)


    def update_model_parameters(self, flat_params):
        idx = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            param.data.copy_(flat_params[idx:idx + num_elements].reshape(param.shape))
            idx += num_elements



    def calculate_gradient(self, F, loss, epsilon=1e-5):
        grad = torch.zeros_like(self.theta)

        for i in range(len(self.theta)):
            original_value = self.theta[i].item()

            #Perturb positively
            self.theta[i] = original_value + epsilon
            loss_plus = loss(F(self.unflatten_parameters(self.theta)))

            #Perturb negatively
            self.theta[i] = original_value - epsilon
            loss_minus = loss(F(self.unflatten_parameters(self.theta)))

            #Approximate derivative
            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

            #Restore original parameter value
            self.theta[i] = original_value

        return grad







