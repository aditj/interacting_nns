import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
### parallelize the mc runs
from multiprocessing import Pool
import tqdm as tqdm
import concurrent.futures

from neural_networks import Net

# Define the function to be estimated
def func(x):
    return 10*(x**2)

N_epochs = 100
N_MC = 100


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


x_range = np.arange(1,1.2,0.2)
N_sample_per_epoch = 10
DO_SINGLE_NN = 0
DO_INTERACTION = 1

lr = 0.003
SGDUPDATES_PER_EPOCH = 1000
LOSS_THRESHOLD = -100

### parallelize the mc runs

def run_mc(mc):
    variances_single = np.zeros((N_epochs,1))
    variances_interaction = np.zeros((N_epochs,2))

    random_seed = mc
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    net_ref = Net()
    net_ref.to(DEVICE)
    optimizer_ref = torch.optim.Adam(net_ref.parameters(), lr=lr)

    var_train = 1

    x_train = np.tile(x_range, N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch*N_epochs).flatten()



    x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)

    y_train = func(x_train) + np.random.normal(0, np.sqrt(var_train), x_train.shape)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    assert x_train.shape == y_train.shape

    x_test = np.linspace(x_range[0],x_range[-1],1).reshape(-1,1)
    x_test = x_range.copy().reshape(-1,1)
    y_test = func(x_test)
    x_test = torch.Tensor(x_test).to(DEVICE)
    y_test = torch.Tensor(y_test).to(DEVICE)

    assert x_test.shape == y_test.shape


    if DO_SINGLE_NN:
        net_single = Net()
        net_single.to(DEVICE)
        loss_fn = net_single.loss
        optimizer_single = torch.optim.SGD(net_single.parameters(), lr=lr)
        net_single.load_state_dict(net_ref.state_dict())

        for epoch in tqdm.tqdm(range(N_epochs)):
            x_batch = x_train[epoch*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch:(epoch+1)*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch].reshape(-1,1)
            y_batch = y_train[epoch*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch:(epoch+1)*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch].reshape(-1,1)
            assert x_batch.shape == y_batch.shape == (N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch,1)
            mean_of_batch = torch.mean(y_batch)
            loss_item = 0
            sgd_index = 0
            while sgd_index < SGDUPDATES_PER_EPOCH and loss_item > LOSS_THRESHOLD:
                optimizer_single.zero_grad()
                loss = loss_fn(x_batch, y_batch)
                loss.backward()
                optimizer_single.step()
                loss_item = loss.item()
                sgd_index += 1
                prediction = net_single.predict(x_test)[0].item()
                # print(f"Single NN: Update {sgd_index+1} of {SGDUPDATES_PER_EPOCH}, loss = {loss_item}, prediction = {prediction}, mean of batch = {mean_of_batch}")

            variances_single[epoch] = net_single.covariance(x_test,func(x_test)).item()
            variance_item = variances_single[epoch]
            prediction = net_single.predict(x_test)
            print(f"Single NN: Epoch {epoch+1} of {N_epochs}, variance = {variance_item}, prediction = {prediction} mean of batch = {mean_of_batch}")
            if variance_item[0] <0.1:
                break
        np.save("variances_single.npy", variances_single)
    if DO_INTERACTION:
        ### Make a 2 level tree with 5 neural networks in the first level and 1 neural network in the second level
        ### each of the 5 neural network recieve 5 samples from the training data
        ### the 5 neural networks in the first level are trained in parallel and the neural network in the second level is trained on the output of the 5 neural networks in the first level
        ### the neural network in the second level recieve 5 noisy sample from the first level neural networks

        first_level_nns = []
        first_level_optimizers = []
        for i in range(N_sample_per_epoch*N_sample_per_epoch):
            first_level_nns.append(Net())
            first_level_nns[i].to(DEVICE)
            first_level_optimizers.append(torch.optim.Adam(first_level_nns[i].parameters(), lr=lr))
            first_level_nns[i].load_state_dict(net_ref.state_dict())
        second_level_nns = []
        second_level_optimizers = []
        for i in range(N_sample_per_epoch):
            second_level_nns.append(Net())
            second_level_nns[i].to(DEVICE)
            second_level_optimizers.append(torch.optim.Adam(second_level_nns[i].parameters(), lr=lr))
            second_level_nns[i].load_state_dict(net_ref.state_dict())
        
        third_level_nn = Net()
        third_level_nn.to(DEVICE)
        third_level_optimizer = torch.optim.Adam(third_level_nn.parameters(), lr=lr)
        third_level_nn.load_state_dict(net_ref.state_dict())

        
        for epoch in tqdm.tqdm(range(N_epochs)):
            x_batch_second_nn = torch.zeros((N_sample_per_epoch,N_sample_per_epoch)).to(DEVICE)
            y_batch_second_nn = torch.zeros((N_sample_per_epoch,N_sample_per_epoch)).to(DEVICE)
            variances_interaction_item = np.zeros((N_sample_per_epoch*N_sample_per_epoch,1))
            for first_level_nn_idx in range(N_sample_per_epoch*N_sample_per_epoch):
                x_batch_first_nn = x_train[epoch*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch+first_level_nn_idx*N_sample_per_epoch:epoch*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch+(first_level_nn_idx+1)*N_sample_per_epoch].reshape(-1,1)
                y_batch_first_nn = y_train[epoch*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch+first_level_nn_idx*N_sample_per_epoch:epoch*N_sample_per_epoch*N_sample_per_epoch*N_sample_per_epoch+(first_level_nn_idx+1)*N_sample_per_epoch].reshape(-1,1)
                assert x_batch_first_nn.shape == y_batch_first_nn.shape == (N_sample_per_epoch,1)


                loss_item = 0
                sgd_index = 0
                while sgd_index < SGDUPDATES_PER_EPOCH//200 and loss_item > LOSS_THRESHOLD:
                    first_level_optimizers[first_level_nn_idx].zero_grad()
                    loss = first_level_nns[first_level_nn_idx].loss(x_batch_first_nn, y_batch_first_nn)
                    loss.backward()
                    first_level_optimizers[first_level_nn_idx].step()
                    loss_item = loss.item()
                    sgd_index += 1
                second_nn_idx = first_level_nn_idx//N_sample_per_epoch
                first_nn_idx = first_level_nn_idx%N_sample_per_epoch
                x_batch_second_nn[second_nn_idx,first_nn_idx] = x_batch_first_nn[0]
                y_batch_second_nn[second_nn_idx,first_nn_idx] = first_level_nns[first_level_nn_idx].sample(x_test).item()
                
                variances_interaction_item[first_level_nn_idx] = first_level_nns[first_level_nn_idx].covariance(x_test,func(x_test)).item()

                variance_item = variances_interaction_item[first_level_nn_idx]
                prediction = first_level_nns[first_level_nn_idx].predict(x_test)
                # print(f"First level {first_level_nn_idx}: Epoch {epoch+1} of {N_epochs}, variance = {variance_item}, prediction = {prediction}")

            variances_interaction[epoch,0] = np.mean(variances_interaction_item)
            assert x_batch_second_nn.shape == y_batch_second_nn.shape == (N_sample_per_epoch,N_sample_per_epoch)

            x_batch_third_nn = torch.zeros((N_sample_per_epoch,1)).to(DEVICE)
            y_batch_third_nn = torch.zeros((N_sample_per_epoch,1)).to(DEVICE)
            for second_level_nn_idx in range(N_sample_per_epoch):
                x_batch_second_nn_curr = x_batch_second_nn[second_level_nn_idx].reshape(-1,1)
                y_batch_second_nn_curr = y_batch_second_nn[second_level_nn_idx].reshape(-1,1)
                assert x_batch_second_nn_curr.shape == y_batch_second_nn_curr.shape == (N_sample_per_epoch,1)
                loss_item = 0
                sgd_index = 0
                while sgd_index < SGDUPDATES_PER_EPOCH//20 and loss_item > LOSS_THRESHOLD:
                    second_level_optimizers[second_level_nn_idx].zero_grad()
                    loss = second_level_nns[second_level_nn_idx].loss(x_batch_second_nn_curr, y_batch_second_nn_curr)
                    loss.backward()
                    second_level_optimizers[second_level_nn_idx].step()
                    loss_item = loss.item()
                    sgd_index += 1
                x_batch_third_nn[second_level_nn_idx] = x_batch_second_nn_curr[0]
                y_batch_third_nn[second_level_nn_idx] = second_level_nns[second_level_nn_idx].sample(x_test).item()

                variances_interaction_item[second_level_nn_idx] = second_level_nns[second_level_nn_idx].covariance(x_test,func(x_test)).item()
                variance_item = variances_interaction_item[second_level_nn_idx]
                prediction = second_level_nns[second_level_nn_idx].predict(x_test)
                # print(f"Second level {second_level_nn_idx}: Epoch {epoch+1} of {N_epochs}, variance = {variance_item}, prediction = {prediction}")

            loss_item = 0
            sgd_index = 0
            while sgd_index < SGDUPDATES_PER_EPOCH and loss_item > LOSS_THRESHOLD:
                third_level_optimizer.zero_grad()
                loss = third_level_nn.loss(x_batch_third_nn, y_batch_third_nn)
                loss.backward()
                third_level_optimizer.step()
                loss_item = loss.item()
                sgd_index += 1

            
            variances_interaction[epoch,1] = third_level_nn.covariance(x_test,func(x_test)).item()
            variance_item = variances_interaction[epoch]
            prediction = third_level_nn.predict(x_test)
            # print(f"Third Level: Epoch {epoch+1} of {N_epochs}, variance = {variance_item}, prediction = {prediction}")

            ## Copy weights from second level to first level
            for first_level_nn_idx in range(N_sample_per_epoch*N_sample_per_epoch):
                first_level_nns[first_level_nn_idx].load_state_dict(third_level_nn.state_dict())
            for second_level_nn_idx in range(N_sample_per_epoch):
                second_level_nns[second_level_nn_idx].load_state_dict(third_level_nn.state_dict())
            if variance_item[0] <0.1 and variance_item[1] < 0.1:
                break
        np.save("variances_interaction_3.npy", variances_interaction)
    return variances_single, variances_interaction
if DO_SINGLE_NN or DO_INTERACTION:
    mcs = range(N_MC)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(run_mc, mcs)))
if DO_SINGLE_NN:
    variances_single = np.zeros((N_MC,N_epochs,1))
    for mc in mcs:
        variances_single[mc], _ = results[mc]
    np.save("variances_single.npy", variances_single)

if DO_INTERACTION:
    variances_interaction = np.zeros((N_MC,N_epochs,2))
    for mc in mcs:
        _, variances_interaction[mc] = results[mc]
    np.save("variances_interaction_3.npy", variances_interaction)
            

variances_interaction = np.load("variances_interaction_3.npy")
variances_single = np.load("variances_single.npy")
variances_two = np.load("variances_interaction.npy")
plt.plot(variances_single, label="Single NN")
plt.plot(variances_two.mean(axis=0)[:,0], label="2 Interacting NN")
plt.plot(variances_interaction.mean(axis=0)[:,0], label="3 Interacting NN")
# plt.plot(variances_interaction[:,1], label="Interaction NN Second Level")
plt.xlabel("Epoch")
plt.ylabel("Covariance")

plt.legend()
plt.savefig("./interacting_nns/plots/variances_with3.png")

