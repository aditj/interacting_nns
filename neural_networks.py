
import torch
import torch.nn as nn
import numpy as np
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define the network to estimate the function as a mean and variance function Gaussian distribution
class Net(nn.Module):
    def __init__(self, n_feature = 1, n_hidden = [4,4], n_output = 1,std = np.sqrt(2)):
        super(Net, self).__init__()
        # Define the structure of the network
        self.n_hidden = n_hidden
        self.input = nn.Linear(n_feature, n_hidden[0])
        for i in range(len(n_hidden)-1):
            setattr(self, 'hidden'+str(i), nn.Linear(n_hidden[i], n_hidden[i+1]))
        self.mu = nn.Linear(n_hidden[-1], n_output)
        # self.sigma = torch.tensor([std]).reshape(1,1).to(DEVICE)
        ## reshape sigma with the same shape as mu
        self.sigma = nn.Linear(n_hidden[-1], n_output).to(DEVICE)
    def forward(self, x):
        # Forward propagation
        x = torch.tanh(self.input(x))
        for i in range(len(self.n_hidden)-1):
            x = torch.tanh(getattr(self, 'hidden'+str(i))(x))
        mu = self.mu(x)
        elu = nn.ELU()
        sigma = elu(self.sigma(x)) + 1 + 1e-6
        # sigma = torch.tensor([self.sigma]).to(DEVICE)
        # ## reshape sigma with the same shape as mu
        # sigma = sigma.repeat(mu.shape[0],1)
        ### ensure sigma is positive
        return torch.cat((mu, sigma), dim = 1)
    def predict(self, x):
        # Predict the mean and variance
        x = self.forward(x)
        return x[:,0], x[:,1]
    def return_sigma(self,x):
        x = self.forward(x)
        return x[:,1]
    def sample(self, x,noise_var = 1):
        # Sample from the Gaussian distribution
        x = self.forward(x)
        mean = x[:,0]
        mean = mean.flatten()
        std = x[:,1]
        noise = torch.randn(x.shape[0]).to(DEVICE)*noise_var
        eps = torch.randn(x.shape[0]).to(DEVICE)
        op = mean  + noise #+ eps*(std**2)
        ### remove grad from output
        return op.detach()
    def covariance(self,x,true_est):
        x = self.forward(x)
        mean = x[:,0]
        std = x[:,1]
        covariance = torch.mean((mean - true_est)**2 + std**2)
        ### return mean of variance of samples from true est
        return covariance
   
# Define the loss function
    def loss(self, x, y):
        mu_est,sigma_est  = self.predict(x)
        sigma_est += np.sqrt(1)
        dist = torch.distributions.normal.Normal(mu_est, sigma_est)
        return -dist.log_prob(y).mean()
    def validation(self, x, y):
        prediction,pred_std = self.predict(x)
        ## print prediction which is distance 2 between prediction and y
        prediction = prediction.reshape(-1,1)
        assert prediction.shape == y.shape
        ### take difference 1d
        
        discp =  (prediction - y)**2 + pred_std**2
        disp = torch.sqrt(discp)
        return discp.mean()
    
class BayesianNeuralNetworkForRegression(nn.Module):
    def __init__(self, n_feature = 1, n_hidden = [6,4], n_output = 1,std = 0.1):
        ### define two networks, mean network and variance network
        super(BayesianNeuralNetworkForRegression, self).__init__()
        # Define the structure of the mean network
        self.n_hidden = n_hidden
        self.input = nn.Linear(n_feature, n_hidden[0])
        for i in range(len(n_hidden)-1):
            setattr(self, 'hidden'+str(i), nn.Linear(n_hidden[i], n_hidden[i+1]))
        self.mu = nn.Linear(n_hidden[-1], n_output)
        # Define the structure of the variance network
        self.input_sigma = nn.Linear(n_feature, n_hidden[0])
        for i in range(len(n_hidden)-1):
            setattr(self, 'hidden_sigma'+str(i), nn.Linear(n_hidden[i], n_hidden[i+1]))
        self.sigma = nn.Linear(n_hidden[-1], n_output)
    def forward(self, x):
        # Forward propagation
        x = torch.tanh(self.input(x))
        for i in range(len(self.n_hidden)-1):
            x = torch.tanh(getattr(self, 'hidden'+str(i))(x))
        mu = self.mu(x)
        x = torch.tanh(self.input_sigma(x))
        for i in range(len(self.n_hidden)-1):
            x = torch.tanh(getattr(self, 'hidden_sigma'+str(i))(x))
        sigma = self.sigma(x)
        return torch.cat((mu, sigma), dim = 1)
    
