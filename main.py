
### 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import tqdm as tqdm



# Define the network to estimate the function as a mean and variance function Gaussian distribution
class Net(nn.Module):
    def __init__(self, n_feature = 1, n_hidden = [12,12], n_output = 1):
        super(Net, self).__init__()
        # Define the structure of the network
        self.n_hidden = n_hidden
        
        self.input = nn.Linear(n_feature, n_hidden[0])
        for i in range(len(n_hidden)-1):
            setattr(self, 'hidden'+str(i), nn.Linear(n_hidden[i], n_hidden[i+1]))
        self.mu = nn.Linear(n_hidden[-1], n_output)
        self.sigma = nn.Linear(n_hidden[-1], n_output)
    def forward(self, x):
        # Forward propagation
        x = torch.tanh(self.input(x))
        elu = nn.ELU()
        for i in range(len(self.n_hidden)-1):
            x = torch.tanh(getattr(self, 'hidden'+str(i))(x))
        mu = self.mu(x)
        sigma = elu(self.sigma(x)) + 1 + 1e-6
        ### ensure sigma is positive
        return torch.cat((mu, sigma), dim = 1)
    def predict(self, x):
        # Predict the mean and variance
        x = self.forward(x)
        return x[:,0], x[:,1]
    def sample(self, x,noise_var = 0.1):
        # Sample from the Gaussian distribution
        x = self.forward(x)
        mean = x[:,0]
        std = x[:,1]
        noise = torch.randn(x.shape[0])*noise_var
        eps = torch.randn(x.shape[0])
        return mean + eps*std + noise
# Define the loss function
    def loss(self, x, y):
        mu_est,sigma_est  = self.predict(x)
        dist = torch.distributions.normal.Normal(mu_est, sigma_est)
        return -dist.log_prob(y).mean()

# Define the function to be estimated
def f(x):
    return x**2-6*x+9 

N_MC = 1
N_epochs = 2000
losses = np.zeros((N_MC,N_epochs))
losses1 = np.zeros((N_MC,N_epochs))
losses2 = np.zeros((N_MC,N_epochs))

sigma_obs_noise = 0.01

N_batches = 100
N_samples = len(np.arange(1,5.2,0.2)) * N_batches
DO_SINGLE_NN = 1
DO_INTERACTION = 1
# Define the training data

for j in tqdm.tqdm(range(N_MC)):
    random_seed = j
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    x_train = np.repeat(np.arange(1,5.2,0.2),N_batches,axis = 0).flatten()
    var_train = 0.01
    y_train = f(x_train).reshape(N_samples,1) + np.random.randn(N_samples,1)*var_train
    assert len(x_train) == len(y_train)
    # Shuffle the data
    idx = np.arange(N_samples)
    idx = np.random.shuffle(idx)

    x_train = x_train[idx].flatten()
    y_train = y_train[idx].flatten()
    # Define the test data
    x_test = np.linspace(1,5.2,100).reshape(-1,1)
    y_test = f(x_test)
    
    if DO_SINGLE_NN:
        # Define the network
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.0003)
        loss_func = net.loss

        # Train the network
        N_sample_per_batch = int(N_samples/N_batches)
        for epoch in tqdm.tqdm(range(N_epochs)):
            for i in range(N_batches):
                x_batch = x_train[i*N_sample_per_batch:i*N_sample_per_batch + N_sample_per_batch//2].reshape(-1,1)
                y_batch = y_train[i*N_sample_per_batch:i*N_sample_per_batch + N_sample_per_batch//2]
                assert len(x_batch) == len(y_batch) == N_sample_per_batch//2
                optimizer.zero_grad()

                loss = loss_func(torch.Tensor(x_batch), torch.Tensor(y_batch))

                loss.backward()
                optimizer.step()
            
            losses[j,epoch] = loss.item()
            #print("Loss: ", loss.item())
        # # Plot the results
        # plt.plot(x_train, y_train, 'o',label = 'train')
        # plt.plot(x_test, y_test, 'k',label = 'test')
        # plt.plot(x_test, net.predict(torch.Tensor(x_test))[0].detach().numpy(), label = 'mean')
        # plt.fill_between(x_test.flatten(), net.predict(torch.Tensor(x_test))[0].detach().numpy() - net.predict(torch.Tensor(x_test))[1].detach().numpy(), net.predict(torch.Tensor(x_test))[0].detach().numpy() + net.predict(torch.Tensor(x_test))[1].detach().numpy(), alpha = 0.5)
        # plt.legend()
        # plt.savefig(f'./interacting_nns/plots/results_{j}.png')
       
    
    ### Two neural networks "interacting"
    #### Each neural network recieves N_batch samples observation from true function f(x)
    #### It also recieves N_batch sampled from the other neural networks current estimate of the function
    #### The two neural networks are trained iteratively
    if DO_INTERACTION:
        # Define the network
        nn1 = Net()
        nn2 = Net()
        optimizer1 = torch.optim.Adam(nn1.parameters(), lr = 0.0003)
        optimizer2 = torch.optim.Adam(nn2.parameters(), lr = 0.0003)
        loss_func1 = nn1.loss
        loss_func2 = nn2.loss
        print("Here")
    
        N_sample_per_batch = int(N_samples/N_batches)
        x_batch_nn1 = x_train[0:N_sample_per_batch].reshape(-1,1)
        y_batch_nn1 = y_train[0:N_sample_per_batch]
        assert len(x_batch_nn1) == len(y_batch_nn1) == N_sample_per_batch
        optimizer1.zero_grad()
        loss1 = loss_func1(torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
        loss1.backward()

        optimizer1.step()
        x_batch_nn2 = x_train[0:N_sample_per_batch].reshape(-1,1)
        y_batch_nn2 = y_train[0:N_sample_per_batch]
        assert len(x_batch_nn2) == len(y_batch_nn2) == N_sample_per_batch
        optimizer2.zero_grad()
        loss2 = loss_func2(torch.Tensor(x_batch_nn2), torch.Tensor(y_batch_nn2))
        loss2.backward()
        optimizer2.step()

        
        for epoch in tqdm.tqdm(range(N_epochs)):
            
            for i in range(N_batches):
                if epoch == 0 and i ==0:
                    continue
            
                ### Input to neural network 1
                x_batch1 = x_train[i*N_sample_per_batch:i*N_sample_per_batch + int(N_sample_per_batch/2)].reshape(-1,1)
                x_batch2 = x_train[i*N_sample_per_batch + int(N_sample_per_batch/2):(i+1)*N_sample_per_batch].reshape(-1,1)
                x_batch_nn1 = np.concatenate((x_batch1, x_batch2), axis = 0)
                y_batch1 = y_train[i*N_sample_per_batch:i*N_sample_per_batch + int(N_sample_per_batch/2)]
                y_batch2 = nn2.sample(torch.Tensor(x_batch2),noise_var=sigma_obs_noise).detach().numpy()
                y_batch_nn1 = np.concatenate((y_batch1, y_batch2), axis = 0)
                
            
                
                ### Input to neural network 2
                x_batch1 = x_train[i*N_sample_per_batch:i*N_sample_per_batch + int(N_sample_per_batch/2)].reshape(-1,1)
                x_batch2 = x_train[i*N_sample_per_batch + int(N_sample_per_batch/2):(i+1)*N_sample_per_batch].reshape(-1,1)
                x_batch_nn2 = np.concatenate((x_batch1, x_batch2), axis = 0)
                y_batch1 = nn1.sample(torch.Tensor(x_batch1),noise_var=sigma_obs_noise).detach().numpy()
                y_batch2 = nn1.sample(torch.Tensor(x_batch2),noise_var=sigma_obs_noise).detach().numpy()
                #y_batch2 = y_train[i*N_sample_per_batch + int(N_sample_per_batch/2):(i+1)*N_sample_per_batch]
                y_batch_nn2 = np.concatenate((y_batch1, y_batch2), axis = 0)
                assert len(x_batch_nn1) == len(y_batch_nn1) == len(x_batch_nn2) == len(y_batch_nn2) == N_sample_per_batch

                ### Train neural network 1
                optimizer1.zero_grad()
                loss1 = loss_func1(torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
                loss1.backward()
                optimizer1.step()
                ### Train neural network 2
                optimizer2.zero_grad()
                loss2 = loss_func2(torch.Tensor(x_batch_nn2), torch.Tensor(y_batch_nn2))
                loss2.backward()
                optimizer2.step()
            
            

            losses1[j,epoch] = loss1.item()
            losses2[j,epoch] = loss2.item()
            print("Loss1: ", loss1.item())
            print("Loss2: ", loss2.item())
        # Save Weights
        torch.save(nn1.state_dict(), f'./interacting_nns/weights/nn1_{j}.pt')
        torch.save(nn2.state_dict(), f'./interacting_nns/weights/nn2_{j}.pt')
        # Plot the results
        # plt.plot(x_train, y_train, 'o',label = 'train')
        # plt.plot(x_test, y_test, 'k',label = 'test')
        # plt.plot(x_test, nn1.predict(torch.Tensor(x_test))[0].detach().numpy(), label = 'mean NN1')
        # plt.plot(x_test, nn2.predict(torch.Tensor(x_test))[0].detach().numpy(), label = 'mean NN2')

        # plt.fill_between(x_test.flatten(), nn1.predict(torch.Tensor(x_test))[0].detach().numpy() - nn1.predict(torch.Tensor(x_test))[1].detach().numpy(), nn1.predict(torch.Tensor(x_test))[0].detach().numpy() + nn1.predict(torch.Tensor(x_test))[1].detach().numpy(), alpha = 0.5)
        # plt.legend()
        # plt.savefig(f'./interacting_nns/plots/results_int_{j}.png')
        # plt.close()
        
if DO_INTERACTION:
    np.save('./interacting_nns/weights/losses1.npy',losses1)
    np.save('./interacting_nns/weights/losses2.npy',losses2)    
if DO_SINGLE_NN:
    np.save('./interacting_nns/weights/losses.npy',losses)

## line with slope -1 and from (1,5) 
x = np.linspace(1,N_epochs,100)
y = 5 - np.log(x)
y_3 = 2.5 - np.log(x)/3


losses = np.load('./interacting_nns/weights/losses.npy')
losses1 = np.load('./interacting_nns/weights/losses1.npy')
losses2 = np.load('./interacting_nns/weights/losses2.npy')
print("LOADED  LOSSES")
print(losses.mean(axis=0),losses1.mean(axis=0),losses2.mean(axis=0))

fig, axs = plt.subplots(1,1,figsize = (5,5))
axs.plot(losses.mean(axis=0),label='Single NN')
axs.plot(losses1.mean(axis=0),label='Interacting NN1')
axs.plot(losses2.mean(axis=0),label='Interacting NN2')
axs.plot(x,y,'k--',label='slope -1')
axs.plot(x,y_3,'k--',label='slope -1/3')
### log scale x
axs.set_xscale('log')
# plot slope -1 line

# plot slope -1/3 line
axs.set_xlabel('Epoch')
axs.set_ylabel('Loss')
axs.set_title(f'Learning process when the public action noise has $\sigma= {sigma_obs_noise}$')
plt.legend()
plt.savefig(f'./interacting_nns/plots/loss_interating_{sigma_obs_noise}.png')


