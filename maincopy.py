
### 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import tqdm as tqdm



# Define the network to estimate the function as a mean and variance function Gaussian distribution
class Net(nn.Module):
    def __init__(self, n_feature = 1, n_hidden = [12,12,10], n_output = 1,std = 0.1):
        super(Net, self).__init__()
        # Define the structure of the network
        self.n_hidden = n_hidden
        self.input = nn.Linear(n_feature, n_hidden[0])
        for i in range(len(n_hidden)-1):
            setattr(self, 'hidden'+str(i), nn.Linear(n_hidden[i], n_hidden[i+1]))
        self.mu = nn.Linear(n_hidden[-1], n_output)
        self.sigma = torch.tensor([std]).reshape(1,1)
        ## reshape sigma with the same shape as mu
        #self.sigma = nn.Linear(n_hidden[-1], n_output)
    def forward(self, x):
        # Forward propagation
        x = torch.tanh(self.input(x))
        for i in range(len(self.n_hidden)-1):
            x = torch.tanh(getattr(self, 'hidden'+str(i))(x))
        mu = self.mu(x)
        #sigma = elu(self.sigma(x)) + 1 + 1e-6
        sigma = torch.tensor([self.sigma])
        ## reshape sigma with the same shape as mu
        sigma = sigma.repeat(mu.shape[0],1)
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
        mean = mean.flatten()
        std = x[:,1]
        noise = torch.randn(x.shape[0])*noise_var
        eps = torch.randn(x.shape[0])
        return mean  + noise + eps*(std**2)
    
# Define the loss function
    def loss(self, x, y):
        mu_est,sigma_est  = self.predict(x)
        dist = torch.distributions.normal.Normal(mu_est, sigma_est)
        return -dist.log_prob(y).mean()
    def validation(self, x, y):
        prediction = self.predict(x)[0]
        ## print prediction which is distance 2 between prediction and y
        prediction = prediction.reshape(-1,1)
        assert prediction.shape == y.shape
        ### take difference 1d
        
        discp =  (prediction - y)**2
        return discp.mean()
# Define the function to be estimated
def f(x):
    return x**2-6*x+9 

N_MC = 1
N_epochs = 4000
losses = np.zeros((N_MC,N_epochs))
losses1 = np.zeros((N_MC,N_epochs))
losses2 = np.zeros((N_MC,N_epochs))
vali_losses = np.zeros((N_MC,N_epochs))
vali_losses1 = np.zeros((N_MC,N_epochs))
vali_losses2 = np.zeros((N_MC,N_epochs))


sigma_obs_noise = 0

N_batches = 10000
x_range = np.arange(1,5.2,0.2)
N_sample_per_batch = 100
N_samples =len(x_range) * N_batches * N_sample_per_batch
N_batches_per_epoch = 100
DO_SINGLE_NN = 0
DO_INTERACTION = 0

single_nn_learning_rate = 0.0003
interacting_nn_learning_rate = 0.003
# Define the training data

for j in tqdm.tqdm(range(N_MC)):
    random_seed = j
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    
    x_train = np.tile(x_range,N_sample_per_batch*N_batches).flatten()

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
    
    # plt.plot(x_train, y_train, 'o',label = 'train')
    # plt.plot(x_test, y_test, 'k',label = 'test')
    # plt.legend()
    # plt.savefig(f'./interacting_nns/plots/data_{j}.png')

    if DO_SINGLE_NN:
        # Define the network
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr = single_nn_learning_rate)
        loss_func = net.loss

        # Train the network
        for epoch in tqdm.tqdm(range(N_epochs)):
            
            batches_for_epoch = np.random.choice(N_batches,N_batches_per_epoch,replace=False)
            len_batches_for_epoch = len(batches_for_epoch)
            for i in range(len_batches_for_epoch):
                x_batch = x_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch].reshape(-1,1)
                y_batch = y_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch]
                assert len(x_batch) == len(y_batch) == N_sample_per_batch
                optimizer.zero_grad()

                loss = loss_func(torch.Tensor(x_batch), torch.Tensor(y_batch))

                loss.backward()
                optimizer.step()

            losses[j,epoch] = loss.item()
            vali_losses[j,epoch] = net.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
            print("Validation Loss: ", vali_losses[j,epoch])
            print("Loss: ", loss.item())
            if loss.item()<-1:
                losses[j,epoch+1:] = loss.item()
                vali_losses[j,epoch+1:] = net.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                break
        # Save Weights
        torch.save(net.state_dict(), f'./interacting_nns/weights/nn_{j}.pt')
        # # Plot the results
        # plt.plot(x_train, y_train, 'o',label = 'train')
        # plt.plot(x_test, y_test, 'k',label = 'test')
        # plt.plot(x_test, net.predict(torch.Tensor(x_test))[0].detach().numpy(), label = 'mean')
        # print(x_test.shape)
        # # plt.fill_between(x_test.flatten(), net.predict(torch.Tensor(x_test)).detach().numpy().reshape(-1,1) - 0.1, net.predict(torch.Tensor(x_test)).detach().numpy() + 0.1, alpha = 0.5)
        # plt.legend()
        # plt.savefig(f'./interacting_nns/plots/results_{j}.png')
       
    
    ### Two neural networks "interacting"
    #### Each neural network recieves N_batch samples observation from true function f(x)
    #### It also recieves N_batch sampled from the other neural networks current estimate of the function
    #### The two neural networks are trained iteratively
    if DO_INTERACTION:
        # Define the network
        nn1 = Net()
        nn2 = Net(std = 0.1)
        optimizer1 = torch.optim.Adam(nn1.parameters(), lr = 0.003)
        optimizer2 = torch.optim.Adam(nn2.parameters(), lr = 0.0003)
        loss_func1 = nn1.loss
        loss_func2 = nn2.loss
    
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

        vali_losses1[j,0] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
        vali_losses2[j,0] = nn2.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
        losses1[j,0] = loss1.item()
        losses2[j,0] = loss2.item()
        
        for epoch in tqdm.tqdm(range(1,N_epochs)):
            
            batches_for_epoch = np.random.choice(N_batches,N_batches_per_epoch,replace=False)
            len_batches_for_epoch = len(batches_for_epoch)
            for i in range(len_batches_for_epoch):
                ### Input to neural network 1
                x_batch_nn1 = x_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch].reshape(-1,1)
                y_batch_nn1 = y_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch]
                ### Input to neural network 2
                
                #assert len(x_batch_nn1) == len(y_batch_nn1) == len(x_batch_nn2) == len(y_batch_nn2) == N_sample_per_batch

                ### Train neural network 1
                optimizer1.zero_grad()
                loss1 = loss_func1(torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
                loss1.backward()
                optimizer1.step()
            ### Train neural network 2
            x_batch_nn2 = np.linspace(x_range[0],x_range[-1],N_sample_per_batch*2).reshape(-1,1)
            y_batch_nn2 = nn1.sample(torch.Tensor(x_batch_nn2),noise_var=sigma_obs_noise).detach().numpy()
            
            optimizer2.zero_grad()
            loss2 = loss_func2(torch.Tensor(x_batch_nn2), torch.Tensor(y_batch_nn2))
            loss2.backward()
            optimizer2.step()
        
                ## Set weights of neural network 1 to neural network 2
            nn1.load_state_dict(nn2.state_dict())
                # ### Randomize nn2 weights
                # nn2 = Net(std = 0.1)

            losses1[j,epoch] = loss1.item()
            if loss1.item()<0 and loss2.item()<0:
                losses1[j,epoch+1:] = loss1.item()
                losses2[j,epoch+1:] = loss2.item()
                break
            losses2[j,epoch] = loss2.item()
            vali_losses1[j,epoch] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
            vali_losses2[j,epoch] = nn2.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
            print("Validation Loss1: ", vali_losses1[j,epoch])
            print("Loss1: ", loss1.item())
            print("Validation Loss2: ", vali_losses2[j,epoch])
            print("Loss2: ", loss2.item())
        # Save Weights
        torch.save(nn1.state_dict(), f'./interacting_nns/weights/nn1_{j}.pt')
        torch.save(nn2.state_dict(), f'./interacting_nns/weights/nn2_{j}.pt')        
if DO_INTERACTION:
    np.save('./interacting_nns/weights/losses1_weight_copy.npy',losses1)
    np.save('./interacting_nns/weights/losses2_weight_copy.npy',losses2)    
    np.save('./interacting_nns/weights/vali_losses1_weight_copy.npy',vali_losses1)
    np.save('./interacting_nns/weights/vali_losses2_weight_copy.npy',vali_losses2)

if DO_SINGLE_NN:
    np.save('./interacting_nns/weights/losses_weight_copy.npy',losses)
    np.save('./interacting_nns/weights/vali_losses_weight_copy.npy',vali_losses)

## line with slope -1 and from (1,5) 
first_n = N_epochs
x = np.linspace(1,first_n,100)
y = 2.25 - np.log(x)
y[y<0] = 0
y_3 = 3 - np.log(x)/3
y_3[y_3<0] = 0

# plt.plot(x_train, y_train, 'o',label = 'train')
# plt.plot(x_test, y_test, 'k',label = 'test')
# plt.plot(x_test, nn.predict(torch.Tensor(x_test))[0].detach().numpy(), label = 'mean')
# plt.legend()
# plt.savefig(f'./interacting_nns/plots/validation_post.png')
# plt.close()
losses = np.load('./interacting_nns/weights/losses_weight_copy.npy')
losses1 = np.load('./interacting_nns/weights/losses1_weight_copy.npy')
losses2 = np.load('./interacting_nns/weights/losses2_weight_copy.npy')

vali_losses = np.load('./interacting_nns/weights/vali_losses_weight_copy.npy')
vali_losses1 = np.load('./interacting_nns/weights/vali_losses1_weight_copy.npy')
vali_losses2 = np.load('./interacting_nns/weights/vali_losses2_weight_copy.npy')

print("Loaded Losses")
print(vali_losses.shape)
plt.plot(vali_losses.mean(axis=0)[0:first_n],label='Single NN')
plt.plot(vali_losses1.mean(axis=0)[0:first_n],label='Interacting NN1')
plt.plot(vali_losses2.mean(axis=0)[0:first_n],label='Interacting NN2')
plt.plot(x,y,'k--',label='slope -1')
plt.plot(x,y_3,'k--',label='slope -1/3')
plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title(f'Validation Loss when the public action noise has $\sigma= {sigma_obs_noise}$')
plt.legend()
plt.savefig(f'./interacting_nns/plots/vali_loss_interating_{sigma_obs_noise}_weight_copy.png')

fig, axs = plt.subplots(1,1,figsize = (5,5))
### Plot first 400 epochs
axs.plot(losses.mean(axis=0)[0:first_n],label='Single NN')
axs.plot(losses1.mean(axis=0)[0:first_n],label='Interacting NN1')
axs.plot(losses2.mean(axis=0)[0:first_n],label='Interacting NN2')
# axs.plot(losses.mean(axis=0),label='Single NN')
# axs.plot(losses1.mean(axis=0),label='Interacting NN1')
# axs.plot(losses2.mean(axis=0),label='Interacting NN2')
# axs.plot(x,y,'k--',label='slope -1')
# axs.plot(x,y_3,'k--',label='slope -1/3')
### log scale x
axs.set_xscale('log')
# plot slope -1 line

# plot slope -1/3 line
axs.set_xlabel('Epoch')
axs.set_ylabel('Loss')
axs.set_title(f'Learning process when the public action noise has $\sigma= {sigma_obs_noise}$')
plt.legend()
plt.savefig(f'./interacting_nns/plots/loss_interating_{sigma_obs_noise}_weight_copy.png')


### Append details of all the parameters and the results to a text file along with time
import datetime
with open('./interacting_nns/plots/parameters.txt','a') as f:
    f.write(f'Run on {datetime.datetime.now()}\n')
    f.write(f'N_MC = {N_MC}\n')
    f.write(f'N_epochs = {N_epochs}\n')
    f.write(f'N_batches = {N_batches}\n')
    f.write(f'N_sample_per_batch = {N_sample_per_batch}\n')
    f.write(f'N_samples = {N_samples}\n')
    f.write(f'N_batches_per_epoch = {N_batches_per_epoch}\n')
    f.write(f'sigma_obs_noise = {sigma_obs_noise}\n')
    f.write(f'x_range = {x_range}\n')
    f.write(f'learning rate of single NN = {single_nn_learning_rate}\n')
    f.write(f'learning rate of interacting NN = {interacting_nn_learning_rate}\n')
