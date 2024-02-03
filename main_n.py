
### 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import tqdm as tqdm

from neural_networks import Net
### 1/k^(2^n-1), 
### -(2^n-1)
# Define the function to be estimated
def f(x):
    return 10*x**2
N_MC = 1
N_epochs = 100
losses = np.zeros((N_MC,N_epochs))
losses1 = np.zeros((N_MC,N_epochs))
vali_losses = np.zeros((N_MC,N_epochs))
vali_losses1 = np.zeros((N_MC,N_epochs))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma_obs_noise = 0

N_batches = 100000
x_range = np.arange(1,1.2,0.2)
N_sample_per_batch = 5
N_samples = len(x_range) * N_batches * N_sample_per_batch
N_batches_per_epoch = 1
DO_SINGLE_NN = 0
DO_INTERACTION = 1

validation_threshold = 0

SGDUPDATES_PER_EPOCH = 40000

LOSS_THRESHOLD = -10

single_nn_learning_rate = 0.0003
interacting_nn_learning_rate = 0.0003
# Define the training data
N_agents_list = [1]



for j in tqdm.tqdm(range(N_MC)):
    random_seed = j
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    ### Initialize Neural Network for reference weights 
    # Define the network
    net_ref = Net().to(DEVICE)
    optimizer_ref = torch.optim.SGD(net_ref.parameters(), lr = single_nn_learning_rate)
    loss_func_ref = net_ref.loss
    
    
    x_train = np.tile(x_range,N_sample_per_batch*N_batches).flatten()
    ### Convert to tensor and transfer to GPU
    x_train = torch.Tensor(x_train).to(DEVICE)
    var_train = 2
    ### torch random randn 
    noise = torch.randn(N_samples,1).to(DEVICE)*var_train
    y_train = f(x_train).reshape(N_samples,1) + noise
    y_train = torch.Tensor(y_train).to(DEVICE)
    assert len(x_train) == len(y_train)
    # Shuffle the data
    idx = np.arange(N_samples)
    # idx = np.random.shuffle(idx)

    x_train = x_train[idx].flatten()
    y_train = y_train[idx].flatten()
    # Define the test data
    x_test = np.linspace(x_range[0],x_range[-1],100).reshape(-1,1)
    x_test = x_range.copy().reshape(-1,1)
    y_test = f(x_test)
    x_test = torch.Tensor(x_test).to(DEVICE)
    y_test = torch.Tensor(y_test).to(DEVICE)
    
    estimates_single_nn = np.zeros((N_epochs,len(x_test)))
    estimates_interacting_nn = np.zeros((len(N_agents_list),N_epochs,len(x_test)))

    variance_single_nn = np.zeros((N_epochs,len(x_test)))
    variance_interacting_nn = np.zeros((len(N_agents_list),N_epochs,len(x_test)))
  
    if DO_SINGLE_NN:
        # Define the network
        net = Net(std = var_train).to(DEVICE)
        optimizer = torch.optim.SGD(net.parameters(), lr = single_nn_learning_rate)
        loss_func = net.loss
        net.load_state_dict(net_ref.state_dict())
        # Train the network
        for epoch in tqdm.tqdm(range(N_epochs)):
            
            batches_for_epoch = np.arange(N_batches)[epoch*N_batches_per_epoch:(epoch+1)*N_batches_per_epoch]

            len_batches_for_epoch = len(batches_for_epoch)

            for i in range(len_batches_for_epoch):
                x_batch = x_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch].reshape(-1,1)
                y_batch = y_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch]
                assert len(x_batch) == len(y_batch) == N_sample_per_batch
                loss_item = 0
                sgd_index = 0
                while loss_item>LOSS_THRESHOLD and sgd_index<SGDUPDATES_PER_EPOCH:
                    sgd_index += 1                    
                    optimizer.zero_grad()
                    loss = loss_func(torch.Tensor(x_batch), torch.Tensor(y_batch))
                    loss.backward()
                    optimizer.step()
                    # if np.abs(loss_item-loss.item())<0.0001:
                    #     break
                    loss_item = loss.item()
                print("Did SGD update ",sgd_index," times")
            losses[j,epoch] = loss.item()
            vali_losses[j,epoch] = net.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
            print("Validation Loss: ", vali_losses[j,epoch])
            print("Loss: ", loss.item())
            if vali_losses[j,epoch].item()<validation_threshold:
                losses[j,epoch+1:] = loss.item()
                vali_losses[j,epoch+1:] = net.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                estimates_single_nn[epoch+1:] = net.predict(torch.Tensor(x_test))[0].detach().numpy()
                break
            estimates_single_nn[epoch] = net.predict(torch.Tensor(x_test))[0].detach().numpy()
            # variance_single_nn[epoch] = net.return_sigma(torch.Tensor(x_test)).detach().numpy()
            variance_single_nn[epoch] = net.covariance(torch.Tensor(x_test),f(x_test)).detach().numpy()
            print("Estimate:",estimates_single_nn[epoch])
            print("Variance ",variance_single_nn[epoch].mean())
            if variance_single_nn[epoch].mean()<1:
                
                break

    ### Two neural networks "interacting"
    #### Each neural network recieves N_batch samples observation from true function f(x)
    #### It also recieves N_batch sampled from the other neural networks current estimate of the function
    #### The two neural networks are trained iteratively
    for N_agents in N_agents_list:
        vali_losses_agents = np.zeros((N_agents,N_MC,N_epochs))
        losses_agents = np.zeros((N_agents,N_MC,N_epochs))
        if DO_INTERACTION:
            # Define the network
            nn1 = Net(std = var_train).to(DEVICE)
            loss_func1 = nn1.loss
            optimizer1 = torch.optim.SGD(nn1.parameters(), lr = interacting_nn_learning_rate)
            nn1.load_state_dict(net_ref.state_dict())

            agents = [Net(std = var_train).to(DEVICE) for i in range(N_agents)]
            for i in range(N_agents):
                agents[i].load_state_dict(net_ref.state_dict())
            optimizers = [torch.optim.SGD(agents[i].parameters(), lr = interacting_nn_learning_rate) for i in range(N_agents)]
            loss_funcs = [agents[i].loss for i in range(N_agents)]
            
            x_batch_nn1 = x_train[0:N_sample_per_batch].reshape(-1,1)
            y_batch_nn1 = y_train[0:N_sample_per_batch]
            assert len(x_batch_nn1) == len(y_batch_nn1) == N_sample_per_batch
            loss1_item = 0
            sgd_index = 0
            while(loss1_item>LOSS_THRESHOLD) and sgd_index<SGDUPDATES_PER_EPOCH:
                sgd_index += 1
                optimizer1.zero_grad()
                loss1 = loss_func1(torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
                loss1.backward()
                optimizer1.step()
                # if np.abs(loss1_item-loss1.item())<0.0001:
                #     break
                loss1_item = loss1.item()
            
            vali_losses1[j,0] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
            losses1[j,0] = loss1.item() 
            # variance_interacting_nn[N_agents-1,0] = nn1.return_sigma(torch.Tensor(x_test)).detach().numpy()
            variance_interacting_nn[N_agents-1,0] = nn1.covariance(torch.Tensor(x_test),f(x_test)).detach().numpy()
            estimates_interacting_nn[N_agents-1,0] = nn1.predict(torch.Tensor(x_test))[0].detach().numpy()
            print("Estimate:",estimates_interacting_nn[N_agents-1,0])
            print("Variance ",variance_interacting_nn[N_agents-1,0].mean())
            ### Train agent 1
            y_batch_nn1 = nn1.sample(torch.Tensor(x_batch_nn1),noise_var=sigma_obs_noise)
            loss_item = 0
            sgd_index = 0
            while loss_item>LOSS_THRESHOLD and sgd_index<SGDUPDATES_PER_EPOCH:
                sgd_index += 1
                optimizers[0].zero_grad()
                loss = loss_funcs[0](torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
                loss.backward()
                optimizers[0].step()
                # if np.abs(loss_item-loss.item())<0.0001:
                #     break
                loss_item = loss.item()
            vali_losses_agents[0,j,0] = agents[0].validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
            losses_agents[0,j,0] = loss_funcs[0](torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1)).item()


            ### Train agents 2 to N
            for i in range(1,N_agents):
                agents[i].load_state_dict(net_ref.state_dict())
                y_batch_nn1 = agents[i-1].sample(torch.Tensor(x_batch_nn1),noise_var=sigma_obs_noise)
                loss_item = 0
                sgd_index = 0
                while loss_item>LOSS_THRESHOLD and sgd_index<SGDUPDATES_PER_EPOCH:
                    sgd_index += 1
                    optimizers[i].zero_grad()
                    loss = loss_funcs[i](torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
                    loss.backward()
                    optimizers[i].step()
                    loss_item = loss.item()

                vali_losses_agents[i,j,0] = agents[i].validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                losses_agents[i,j,0] = loss_funcs[i](torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1)).item()
            x_batch_nn2 = np.linspace(x_range[0],x_range[-1],N_sample_per_batch*2).reshape(-1,1)
            print("Intialized Neural Networks")
            for epoch in tqdm.tqdm(range(1,N_epochs)):
                x_batch_nn2 = np.tile(x_range,N_sample_per_batch).reshape(-1,1)
                x_batch_nn2 = torch.Tensor(x_batch_nn2).to(DEVICE)
                batches_for_epoch = np.arange(N_batches)[epoch*N_batches_per_epoch:(epoch+1)*N_batches_per_epoch]
                len_batches_for_epoch = len(batches_for_epoch)
                for i in range(len_batches_for_epoch):
                    ### Input to neural network 1
                    x_batch_nn1 = x_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch].reshape(-1,1)
                    y_batch_nn1 = y_train[batches_for_epoch[i]*N_sample_per_batch:batches_for_epoch[i]*N_sample_per_batch + N_sample_per_batch]
                    ### Input to neural network 2
                    #assert len(x_batch_nn1) == len(y_batch_nn1) == len(x_batch_nn2) == len(y_batch_nn2) == N_sample_per_batch

                    ### Train neural network 1
                    loss1_item = 0
                    sgd_index = 0

                    while(loss1_item>LOSS_THRESHOLD) and sgd_index<SGDUPDATES_PER_EPOCH:
                        sgd_index += 1
                        optimizer1.zero_grad()
                        loss1 = loss_func1(torch.Tensor(x_batch_nn1), torch.Tensor(y_batch_nn1))
                        loss1.backward()
                        optimizer1.step()
                        # if np.abs(loss1_item-loss1.item())<0.001:
                        #     break
                        loss1_item = loss1.item()
                    print("Did SGD update ",sgd_index," times")
                losses1[j,epoch] = loss1.item()
                vali_losses1[j,epoch] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()

                ### Train neural network 2
                y_batch_nn2 = nn1.sample(torch.Tensor(x_batch_nn2),noise_var=sigma_obs_noise)
                loss_item = 0
                sgd_index = 0
                while loss_item>LOSS_THRESHOLD  and sgd_index<SGDUPDATES_PER_EPOCH:
                    sgd_index += 1
                    optimizers[0].zero_grad()
                    loss_agent = loss_funcs[0](torch.Tensor(x_batch_nn2), torch.Tensor(y_batch_nn2))
                    loss_agent.backward()
                    optimizers[0].step()
                    # if np.abs(loss_item-loss_agent.item())<0.001:
                    #     break
                    loss_item = loss_agent.item()
                print("Did SGD update ",sgd_index," times")
                losses_agents[0,j,epoch] = loss_agent.item()
                vali_losses_agents[0,j,epoch] = agents[0].validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()

                ### Train agents 3 to N-1
                
                for i in range(1,N_agents):
                    y_batch_nn2 = agents[i-1].sample(torch.Tensor(x_batch_nn2),noise_var=sigma_obs_noise)
                    loss_item = 0
                    sgd_index = 0
                    while loss_item>LOSS_THRESHOLD and sgd_index<SGDUPDATES_PER_EPOCH:
                        sgd_index += 1
                        optimizers[i].zero_grad()
                        loss_agent = loss_funcs[i](torch.Tensor(x_batch_nn2), torch.Tensor(y_batch_nn2))
                        loss_agent.backward()
                        optimizers[i].step()
                        # if np.abs(loss_item-loss_agent.item())<0.001:
                        #     break
                        loss_item = loss_agent.item()
                    losses_agents[i,j,epoch] = loss_agent.item()
                    vali_losses_agents[i,j,epoch] = agents[i].validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                    print("Did SGD update ",sgd_index," times")
                
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
                if vali_losses1[j,epoch]<validation_threshold:
                    if (vali_losses_agents[:,j,epoch]<validation_threshold).sum() == N_agents:
                        losses1[j,epoch+1:] = loss1.item()
                        losses_agents[i,j,epoch+1:] = loss.item()
                        vali_losses1[j,epoch+1:] = nn1.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item()
                        vali_losses_agents[i,j,epoch+1:] = net.validation(torch.Tensor(x_test), torch.Tensor(y_test)).item() 
                        variance_interacting_nn[N_agents-1,epoch:] = nn1.return_sigma(torch.Tensor(x_test)).detach().numpy()  
                        break
                estimates_interacting_nn[N_agents-1,epoch] = nn1.predict(torch.Tensor(x_test))[0].detach().numpy()
                # variance_interacting_nn[N_agents-1,epoch] = nn1.return_sigma(torch.Tensor(x_test)).detach().numpy()
                variance_interacting_nn[N_agents-1,epoch] = nn1.covariance(torch.Tensor(x_test),f(x_test)).detach().numpy()
                print("Variance ",variance_interacting_nn[N_agents-1,epoch].mean())
                print("Estimate:",estimates_interacting_nn[N_agents-1,epoch])
            # Save Weights
                if variance_interacting_nn[N_agents-1,epoch].mean() < 0.1 and epoch>10:
                    break 
            torch.save(nn1.state_dict(), f'./interacting_nns/weights/nn1_{j}.pt')
        
        if DO_INTERACTION:
            np.save(f'interacting_nns/weights/losses1_{N_agents}_weight.npy',losses1)
            np.save(f'./interacting_nns/weights/losses_agents_{N_agents}_weight.npy',losses_agents)   
            np.save(f'./interacting_nns/weights/vali_losses1_{N_agents}_weight.npy',vali_losses1)
            np.save(f'./interacting_nns/weights/vali_losses_agents_{N_agents}_weight.npy',vali_losses_agents)
if DO_SINGLE_NN:
    np.save('./interacting_nns/weights/losses_weight.npy',losses)
    np.save('./interacting_nns/weights/vali_losses_weight.npy',vali_losses)
    np.save('./interacting_nns/weights/estimates_single_nn_weight.npy',estimates_single_nn)
    np.save('./interacting_nns/weights/variance_single_nn_weight.npy',variance_single_nn)
if DO_INTERACTION:
    np.save('./interacting_nns/weights/estimates_interacting_nn_weight.npy',estimates_interacting_nn)
    np.save('./interacting_nns/weights/variance_interacting_nn_weight.npy',variance_interacting_nn)

### Plot estimates of the function
fig,axs = plt.subplots(1,1,figsize = (10,10))
epochs = np.arange(N_epochs)
y_epochs = f(x_test)
# Make y_epochs the same shape as estimates by repeating each entry N_epochs times
y_epochs = np.tile(y_epochs,(N_epochs,1)).reshape(N_epochs,-1)
axs.plot(epochs, y_epochs, 'k',label = 'True Function')
estimates_single_nn = np.load('./interacting_nns/weights/estimates_single_nn_weight.npy')
axs.plot(epochs, estimates_single_nn, label = 'Single NN')

estimates_interacting_nn = np.load('./interacting_nns/weights/estimates_interacting_nn_weight.npy')
for i in range(len(N_agents_list)):
    axs.plot(epochs, estimates_interacting_nn[i,:], label = f'{i+2} Interacting NN')
axs.legend()
axs.set_xlabel('Epoch')
# axs.set_xscale('log')
plt.savefig(f'./interacting_nns/plots/estimates_interating_{sigma_obs_noise}_weight.png')

## Plot variance
fig,axs = plt.subplots(1,1,figsize = (10,10))
epochs = np.arange(N_epochs)
variance_single_nn = np.load('./interacting_nns/weights/variance_single_nn_weight.npy')
axs.plot(epochs, variance_single_nn.mean(axis=1), label = 'Single NN')
variance_interacting_nn = np.load('./interacting_nns/weights/variance_interacting_nn_weight.npy')

for i in range(len(N_agents_list)):
    axs.plot(epochs, variance_interacting_nn[i,:].mean(axis=1), label = f'{i+2} Interacting NN')
axs.legend()
axs.set_xlabel('Epoch')
# axs.set_xscale('log')
plt.savefig(f'./interacting_nns/plots/variance_interating_{sigma_obs_noise}_weight.png')

losses = np.load('./interacting_nns/weights/losses_weight.npy')
vali_losses = np.load('./interacting_nns/weights/vali_losses_weight.npy')

fig_all_losses,axs_all_losses = plt.subplots(1,1,figsize = (10,10))
axs_all_losses.plot(losses.mean(axis=0),label='Single NN')

fig_all,axs_all = plt.subplots(1,1,figsize = (10,10))
axs_all.plot(vali_losses.mean(axis=0),label='Single NN')





l = 0
for N_agents in N_agents_list:
    print("Plotting Losses for N_agents = ",N_agents)
    ## line with slope -1 and from (1,5) 
    first_n = N_epochs

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

    plt.xscale('log')
    # plot slope -1 line

    # plot slope -1/3 line
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning process when the public action noise has $\sigma= {sigma_obs_noise}$')
    plt.legend()
    plt.savefig(f'./interacting_nns/plots/loss_interating_{sigma_obs_noise}_weight_{N_agents}.png')
    plt.close()

    axs_all_losses.plot(np.arange(N_epochs),losses1.flatten(),label=f'Interacting with {N_agents+1} agents')
    axs_all.plot(np.arange(N_epochs),vali_losses1.flatten(),label=f'Interacting with {N_agents+1} agents')

    

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
