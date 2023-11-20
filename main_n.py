
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
    
### 1/k^(2^n-1), 
### -(2^n-1)
# Define the function to be estimated
def f(x):
    return x**2-6*x+9 

N_MC = 1
N_epochs = 1000
losses = np.zeros((N_MC,N_epochs))
losses1 = np.zeros((N_MC,N_epochs))
vali_losses = np.zeros((N_MC,N_epochs))
vali_losses1 = np.zeros((N_MC,N_epochs))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma_obs_noise = 0

N_batches = 10000
x_range = np.arange(1,1.2,0.2)
N_sample_per_batch = 100
N_samples = len(x_range) * N_batches * N_sample_per_batch
N_batches_per_epoch = 100
DO_SINGLE_NN = 1
DO_INTERACTION = 1

single_nn_learning_rate = 0.0003
interacting_nn_learning_rate = 0.0003
# Define the training data
N_agents_list = [1,2,3]





for j in tqdm.tqdm(range(N_MC)):
    random_seed = j
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    ### Initialize Neural Network for reference weights 
    # Define the network
    net_ref = Net().to(DEVICE)
    optimizer_ref = torch.optim.Adam(net_ref.parameters(), lr = single_nn_learning_rate)
    loss_func_ref = net_ref.loss
    
    
    x_train = np.tile(x_range,N_sample_per_batch*N_batches).flatten()
    ### Convert to tensor and transfer to GPU
    x_train = torch.Tensor(x_train).to(DEVICE)
    var_train = 0.01
    y_train = f(x_train).reshape(N_samples,1) + np.random.randn(N_samples,1)*var_train
    y_train = torch.Tensor(y_train).to(DEVICE)
    assert len(x_train) == len(y_train)
    # Shuffle the data
    idx = np.arange(N_samples)
    idx = np.random.shuffle(idx)

    x_train = x_train[idx].flatten()
    y_train = y_train[idx].flatten()
    # Define the test data
    x_test = np.linspace(x_range[0],x_range[-1],100).reshape(-1,1)
    y_test = f(x_test)
    x_test = torch.Tensor(x_test).to(DEVICE)
    y_test = torch.Tensor(y_test).to(DEVICE)
    
    # plt.plot(x_train, y_train, 'o',label = 'train')
    # plt.plot(x_test, y_test, 'k',label = 'test')
    # plt.legend()
    # plt.savefig(f'./interacting_nns/plots/data_{j}.png')

    if DO_SINGLE_NN:
        # Define the network
        net = Net().to(DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), lr = single_nn_learning_rate)
        loss_func = net.loss
        net.load_state_dict(net_ref.state_dict())
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
    for N_agents in N_agents_list:
        vali_losses_agents = np.zeros((N_agents,N_MC,N_epochs))
        losses_agents = np.zeros((N_agents,N_MC,N_epochs))
        if DO_INTERACTION:
            # Define the network
            nn1 = Net().to(DEVICE)
            loss_func1 = nn1.loss
            optimizer1 = torch.optim.Adam(nn1.parameters(), lr = interacting_nn_learning_rate)
                
            nn1.load_state_dict(net_ref.state_dict())


            agents = [Net(std = 0.1).to(DEVICE) for i in range(N_agents)]
            optimizers = [torch.optim.Adam(agents[i].parameters(), lr = interacting_nn_learning_rate) for i in range(N_agents)]
            loss_funcs = [agents[i].loss for i in range(N_agents)]
            
            x_batch_nn1 = x_train[0:N_sample_per_batch].reshape(-1,1)
            y_batch_nn1 = y_train[0:N_sample_per_batch]

            assert len(x_batch_nn1) == len(y_batch_nn1) == N_sample_per_batch
            optimizer1.zero_grad()
            loss1 = loss_func1(torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
            loss1.backward()
            optimizer1.step()
            vali_losses1[j,0] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
            losses1[j,0] = loss1.item() 

            for i in range(N_agents):
                agents[i].load_state_dict(net_ref.state_dict())
                optimizers[i].zero_grad()
                loss = loss_funcs[i](torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
                loss.backward()
                optimizers[i].step()
                vali_losses_agents[i,j,0] = agents[i].validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                losses_agents[i,j,0] = loss_funcs[i](torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1)).item()
            
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
                
                losses1[j,epoch] = loss1.item()
                vali_losses1[j,epoch] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()

                ### Train neural network 2
                x_batch_nn2 = np.linspace(x_range[0],x_range[-1],N_sample_per_batch*2).reshape(-1,1)
                y_batch_nn2 = nn1.sample(torch.Tensor(x_batch_nn2),noise_var=sigma_obs_noise).detach().numpy()
                optimizers[0].zero_grad()
                loss_agent = loss_funcs[0](torch.Tensor(x_batch_nn2), torch.Tensor(y_batch_nn2))
                loss_agent.backward()
                optimizers[0].step()
                losses_agents[0,j,epoch] = loss_agent.item()
                vali_losses_agents[0,j,epoch] = agents[0].validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()

                ### Train agents 3 to N-1
                for i in range(1,N_agents):
                    x_batch_nn2 = np.linspace(x_range[0],x_range[-1],N_sample_per_batch*2).reshape(-1,1)
                    y_batch_nn2 = agents[i-1].sample(torch.Tensor(x_batch_nn2),noise_var=sigma_obs_noise).detach().numpy()
                    optimizers[i].zero_grad()
                    loss_agent = loss_funcs[i](torch.Tensor(x_batch_nn2), torch.Tensor(y_batch_nn2))
                    loss_agent.backward()
                    optimizers[i].step()
                    losses_agents[i,j,epoch] = loss_agent.item()
                    vali_losses_agents[i,j,epoch] = agents[i].validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                
                
                ## Set weights of neural network 1 to N-1 to neural network N

                nn1.load_state_dict(agents[-1].state_dict())
                for i in range(N_agents-1):
                    agents[i].load_state_dict(agents[-1].state_dict())

                ### Output loss                
                print("Loss1: ", loss1.item())
                print("Validation Loss1: ", vali_losses1[j,epoch])
                for i in range(0,N_agents):
                    print(f"Loss{i+2}: ", losses_agents[i,j,epoch])
                    print(f"Validation Loss{i+2}: ", vali_losses_agents[i,j,epoch])
                if loss1.item()<-1:
                    if (losses_agents[:,j,epoch]<0).sum() == N_agents:
                        losses1[j,epoch+1:] = loss1.item()
                        losses_agents[i,j,epoch+1:] = loss.item()
                        vali_losses1[j,epoch+1:] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                        vali_losses_agents[i,j,epoch+1:] = net.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()        
                        break
            # Save Weights
            torch.save(nn1.state_dict(), f'./interacting_nns/weights/nn1_{j}.pt')
        
        if DO_INTERACTION:
            np.save(f'interacting_nns/weights/losses1_{N_agents}_weight.npy',losses1)
            np.save(f'./interacting_nns/weights/losses_agents_{N_agents}_weight.npy',losses_agents)   
            np.save(f'./interacting_nns/weights/vali_losses1_{N_agents}_weight.npy',vali_losses1)
            np.save(f'./interacting_nns/weights/vali_losses_agents_{N_agents}_weight.npy',vali_losses_agents)

if DO_SINGLE_NN:
    np.save('./interacting_nns/weights/losses_weight.npy',losses)
    np.save('./interacting_nns/weights/vali_losses_weight.npy',vali_losses)

losses = np.load('./interacting_nns/weights/losses_weight.npy')
vali_losses = np.load('./interacting_nns/weights/vali_losses_weight.npy')

fig_all_losses,axs_all_losses = plt.subplots(1,1,figsize = (10,10))
axs_all_losses.plot(losses.mean(axis=0),label='Single NN')

fig_all,axs_all = plt.subplots(1,1,figsize = (10,10))
axs_all.plot(vali_losses.mean(axis=0),label='Single NN')

start_x = 1
end_x = np.zeros((len(N_agents_list),))
start_y = np.zeros((len(N_agents_list),))
end_y = 1000

l = 0
for N_agents in N_agents_list:
    print("Plotting Losses for N_agents = ",N_agents)
    ## line with slope -1 and from (1,5) 
    first_n = N_epochs
    

    # plt.plot(x_train, y_train, 'o',label = 'train')
    # plt.plot(x_test, y_test, 'k',label = 'test')
    # plt.plot(x_test, nn.predict(torch.Tensor(x_test))[0].detach().numpy(), label = 'mean')
    # plt.legend()
    # plt.savefig(f'./interacting_nns/plots/validation_post.png')
    # plt.close()
    losses1 = np.load(f'./interacting_nns/weights/losses1_{N_agents}_weight.npy')
    losses2 = np.load(f'./interacting_nns/weights/losses_agents_{N_agents}_weight.npy')

    vali_losses1 = np.load(f'./interacting_nns/weights/vali_losses1_{N_agents}_weight.npy')
    vali_losses_agents = np.load(f'./interacting_nns/weights/vali_losses_agents_{N_agents}_weight.npy')
    
    print("Loaded Losses")
    plt.figure(figsize = (15,15))
    plt.plot(vali_losses.mean(axis=0)[0:first_n],label='Single NN')
    plt.plot(vali_losses1.mean(axis=0)[0:first_n],label='Interacting NN1')
    for j in range(N_agents):
        plt.plot(vali_losses_agents.mean(axis=1)[j,0:first_n],label=f'Interacting NN{j}')

    plt.xscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss when the public action noise has $\sigma= {sigma_obs_noise}$')
    plt.legend()
    plt.savefig(f'./interacting_nns/plots/vali_loss_interating_{sigma_obs_noise}_weight_{N_agents}.png')
    plt.close()


    plt.figure(figsize = (15,15))
    ### Plot first 400 epochs
    plt.plot(losses.mean(axis=0)[0:first_n],label='Single NN')
    plt.plot(losses1.mean(axis=0)[0:first_n],label='Interacting NN1')
    for j in range(N_agents):
        plt.plot(losses_agents.mean(axis=1)[j,0:first_n],label=f'Interacting NN {j+2}')

    # axs.plot(losses.mean(axis=0),label='Single NN')
    # axs.plot(losses1.mean(axis=0),label='Interacting NN1')
    # axs.plot(losses2.mean(axis=0),label='Interacting NN2')
    # axs.plot(x,y,'k--',label='slope -1')
    # axs.plot(x,y_3,'k--',label='slope -1/3')
    ### log scale x
    plt.xscale('log')
    # plot slope -1 line

    # plot slope -1/3 line
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning process when the public action noise has $\sigma= {sigma_obs_noise}$')
    plt.legend()
    plt.savefig(f'./interacting_nns/plots/loss_interating_{sigma_obs_noise}_weight_{N_agents}.png')
    plt.close()

    axs_all_losses.plot(np.arange(N_epochs),losses1.flatten(),label=f'Interacting NN {N_agents}')
    axs_all.plot(np.arange(N_epochs),vali_losses1.flatten(),label=f'Interacting NN {N_agents}')

    end_x[l] = np.argwhere(vali_losses1.flatten()<end_y)[0]
    start_y[l] = vali_losses1.flatten()[start_x]
    l+=1
    ### plot straight line from start to end
    x_line = np.linspace(start_x,end_x[l-1],1000)
    slope = (start_y[l-1]-end_y)/(np.log(end_x[l-1])-np.log(start_x))
    y_line = start_y[l-1] + (np.log(start_x) - np.log(x_line))*slope
    y_line[y_line<0] = 0
    axs_all.plot(x_line,y_line,'k--')
    # Put text in middle of line tilted along the line_x)
    text_x = (np.log(start_x)+np.log(end_x[l-1]))/2
    text_y = start_y[l-1] + (np.log(start_x) - text_x)*slope
    print(vali_losses1.flatten()[0])
    string_slope = f'Slope = {slope:.2f}'
    print(text_x,text_y,string_slope)
    axs_all.text(np.exp(text_x),text_y,string_slope,rotation = -np.arctan(slope)*180/np.pi)



axs_all_losses.plot(losses.mean(axis=0),label='Single NN')
axs_all_losses.set_xscale('log')
axs_all_losses.set_xlabel('Epoch')
axs_all_losses.set_ylabel('Loss')
axs_all_losses.set_title(f'Loss with Different Number of Agents (NNs)')
axs_all_losses.legend()
fig_all_losses.savefig(f'./interacting_nns/plots/loss_interating_{sigma_obs_noise}_weight_all.png')

axs_all.set_xscale('log')
axs_all.set_xlabel('Epoch')
axs_all.set_ylabel('Validation Loss')
axs_all.set_title(f'Validation Loss with Different Number of Agents (NNs)')
axs_all.legend()
fig_all.savefig(f'./interacting_nns/plots/vali_loss_interating_{sigma_obs_noise}_weight_all.png')

### Append details of all the parameters and the results to a text file along with time
import datetime
with open('./interacting_nns/plots/parameters.txt','a') as f:
    f.write(f'Run on {datetime.datetime.now()}\n')
    f.write(f'N_agents = {N_agents}\n')
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
