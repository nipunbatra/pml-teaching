import torch
import hamiltorch
import torch.distributions as dist
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Using a neural network with HMC


class Net_Classification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 4)
        self.fc2 = torch.nn.Linear(4, 2)
        self.fc3 = torch.nn.Linear(2, 1)


    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        logits = self.fc3(x)
        return logits
    
net_classification = Net_Classification().to(device)

### Bayesian Logistic Regression

from sklearn.datasets import make_moons

# Generate data
x, y = make_moons(n_samples=1000, noise=0.1, random_state=0)

x_moon = torch.tensor(x).float().to(device)
y_moon = torch.tensor(y).float().to(device)

def log_prior(theta):
    return dist.Normal(0, 1).log_prob(theta).sum()


def log_likelihood(theta):
    params_list = hamiltorch.util.unflatten(net_classification, theta)

    ## Functional call
    params = net_classification.state_dict()
    for i, (name, _) in enumerate(params.items()):
        params[name] = params_list[i]
    y_pred = torch.func.functional_call(net_classification, params, x_moon).squeeze()
    return dist.Bernoulli(logits=y_pred).log_prob(y_moon).sum()


def log_joint(theta):
    log_prior_val = log_prior(theta)
    log_likelihood_val = log_likelihood(theta)
    log_joint = log_prior_val + log_likelihood_val
    return log_joint



if __name__ == "__main__":
    params_hmc = run_hmc(log_joint, torch.tensor([0.2, 0.5]), 100, 0.1, 5)
    

