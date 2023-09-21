import torch
import hamiltorch
import torch.distributions as dist
import matplotlib.pyplot as plt

# Using a neural network with HMC


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
net = Net()

torch.manual_seed(123)
x_lin = torch.linspace(-3, 3, 90)
theta_0_true = torch.tensor([2.0])
theta_1_true = torch.tensor([3.0])
f = lambda x: theta_0_true + theta_1_true * x
eps = torch.randn_like(x_lin) * 1.0
y_lin = f(x_lin) + eps

def log_prior(theta):
    return dist.Normal(0, 1).log_prob(theta).sum()


def log_likelihood(theta):
    params_list = hamiltorch.util.unflatten(net, theta)

    ## Functional call
    params = net.state_dict()
    for i, (name, _) in enumerate(params.items()):
        params[name] = params_list[i]
    y_pred = torch.func.functional_call(net, params, x_lin.unsqueeze(1)).squeeze()
    return dist.Normal(y_pred, 1).log_prob(y_lin).sum()


def log_joint(theta):
    log_prior_val = log_prior(theta)
    log_likelihood_val = log_likelihood(theta)
    log_joint = log_prior_val + log_likelihood_val
    # print(log_joint, log_prior_val, log_likelihood_val)
    return log_joint



if __name__ == "__main__":
    params_hmc = run_hmc(log_joint, torch.tensor([0.2, 0.5]), 100, 0.1, 5)
    

