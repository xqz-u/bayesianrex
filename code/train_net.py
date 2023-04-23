import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, ENCODING_DIMS, ACTION_DIMS):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 16, 3, stride=1)
        intermediate_dimension = min(784, max(64, ENCODING_DIMS*2))
        self.fc1 = nn.Linear(784, intermediate_dimension)
        self.fc_mu = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc_var = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc2 = nn.Linear(ENCODING_DIMS, 1)
        self.reconstruct1 = nn.Linear(ENCODING_DIMS, intermediate_dimension)
        self.reconstruct2 = nn.Linear(intermediate_dimension, 1568)
        self.reconstruct_conv1 = nn.ConvTranspose2d(2, 4, 3, stride=1)
        self.reconstruct_conv2 = nn.ConvTranspose2d(4, 16, 6, stride=1)
        self.reconstruct_conv3 = nn.ConvTranspose2d(16, 16, 7, stride=2)
        self.reconstruct_conv4 = nn.ConvTranspose2d(16, 4, 10, stride=1)
        self.temporal_difference1 = nn.Linear(ENCODING_DIMS*2, 1, bias=False)#ENCODING_DIMS)
        #self.temporal_difference2 = nn.Linear(ENCODING_DIMS, 1)
        self.inverse_dynamics1 = nn.Linear(ENCODING_DIMS*2, ACTION_DIMS, bias=False) #ENCODING_DIMS)
        #self.inverse_dynamics2 = nn.Linear(ENCODING_DIMS, ACTION_SPACE_SIZE)
        self.forward_dynamics1 = nn.Linear(ENCODING_DIMS + ACTION_DIMS, ENCODING_DIMS, bias=False)# (ENCODING_DIMS + ACTION_SPACE_SIZE) * 2)
        #self.forward_dynamics2 = nn.Linear((ENCODING_DIMS + ACTION_SPACE_SIZE) * 2, (ENCODING_DIMS + ACTION_SPACE_SIZE) * 2)
        #self.forward_dynamics3 = nn.Linear((ENCODING_DIMS + ACTION_SPACE_SIZE) * 2, ENCODING_DIMS)
        self.normal = tdist.Normal(0, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.ACTION_DIMS = ACTION_DIMS
        print("Intermediate dimension calculated to be: " + str(intermediate_dimension))

    def reparameterize(self, mu, var): #var is actually the log variance
        if self.training:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            std = var.mul(0.5).exp()
            eps = self.normal.sample(mu.shape).to(device)
            return eps.mul(std).add(mu)
        else:
            return mu


    def cum_return(self, traj):
        #print("input shape of trajectory:")
        #print(traj.shape)
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.reshape((x.shape[0], 784))
        x = F.leaky_relu(self.fc1(x))
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        z = self.reparameterize(mu, var)

        r = self.fc2(z)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards, mu, var, z

    def estimate_temporal_difference(self, z1, z2):
        x = self.temporal_difference1(torch.cat((z1, z2), 1))
        #x = self.temporal_difference2(x)
        return x

    def forward_dynamics(self, z1, actions):
        x = torch.cat((z1, actions), dim=1)
        x = self.forward_dynamics1(x)
        #x = F.leaky_relu(self.forward_dynamics2(x))
        #x = self.forward_dynamics3(x)
        return x

    def estimate_inverse_dynamics(self, z1, z2):
        concatenation = torch.cat((z1, z2), 1)
        x = self.inverse_dynamics1(concatenation)
        #x = F.leaky_relu(self.inverse_dynamics2(x))
        return x

    def decode(self, encoding):
        x = F.leaky_relu(self.reconstruct1(encoding))
        x = F.leaky_relu(self.reconstruct2(x))
        x = x.view(-1, 2, 28, 28)
        #print("------decoding--------")
        #print(x.shape)
        x = F.leaky_relu(self.reconstruct_conv1(x))
        #print(x.shape)
        x = F.leaky_relu(self.reconstruct_conv2(x))
        #print(x.shape)
        #print(x.shape)
        x = F.leaky_relu(self.reconstruct_conv3(x))
        #print(x.shape)
        #print(x.shape)
        x = self.sigmoid(self.reconstruct_conv4(x))
        #print(x.shape)
        #print("------end decoding--------")
        return x.permute(0, 2, 3, 1)

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mu1, var1, z1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2, var2, z2 = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, z1, z2, mu1, mu2, var1, var2



def reconstruction_loss(decoded, target, mu, logvar):
    num_elements = decoded.numel()
    target_num_elements = decoded.numel()
    if num_elements != target_num_elements:
        print("ELEMENT SIZE MISMATCH IN RECONSTRUCTION")
        sys.exit()
    bce = F.binary_cross_entropy(decoded, target)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= num_elements
    #print("bce: " + str(bce) + " kld: " + str(kld))
    return bce + kld

# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, training_times, training_actions, num_iter, l1_reg, checkpoint_dir, loss_fn):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    temporal_difference_loss = nn.MSELoss()
    inverse_dynamics_loss = nn.CrossEntropyLoss()
    forward_dynamics_loss = nn.MSELoss()

    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs, training_times, training_actions))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels, training_times_sub, training_actions_sub = zip(*training_data)
        validation_split = 1.0
        for i in tqdm(range(len(training_labels)), mininterval=1):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            times_i, times_j = training_times_sub[i]
            actions_i, actions_j = training_actions_sub[i]

            traj_i, traj_j = np.array(traj_i), np.array(traj_j)
            traj_i, traj_j = torch.from_numpy(traj_i).float().to(device), torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)
            num_frames_i, num_frames_j = len(traj_i), len(traj_j)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards, z1, z2, mu1, mu2, logvar1, logvar2 = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)

            decoded1 = reward_network.decode(z1)
            decoded2 = reward_network.decode(z2)
            reconstruction_loss_1 = 10*reconstruction_loss(decoded1, traj_i, mu1, logvar1)
            reconstruction_loss_2 = 10*reconstruction_loss(decoded2, traj_j, mu2, logvar2)

            t1_i, t2_i = np.random.randint(0, len(times_i)), np.random.randint(0, len(times_i))
            t1_j, t2_j = np.random.randint(0, len(times_j)), np.random.randint(0, len(times_j))

            est_dt_i = reward_network.estimate_temporal_difference(mu1[t1_i].unsqueeze(0), mu1[t2_i].unsqueeze(0))
            est_dt_j = reward_network.estimate_temporal_difference(mu2[t1_j].unsqueeze(0), mu2[t2_j].unsqueeze(0))
            real_dt_i = (times_i[t2_i] - times_i[t1_i])/100.0
            real_dt_j = (times_j[t2_j] - times_j[t1_j])/100.0

            actions_1 = reward_network.estimate_inverse_dynamics(mu1[0:-1], mu1[1:])
            actions_2 = reward_network.estimate_inverse_dynamics(mu2[0:-1], mu2[1:])
            target_actions_1 = torch.LongTensor(actions_i[1:]).to(device)
            target_actions_2 = torch.LongTensor(actions_j[1:]).to(device)

            inverse_dynamics_loss_1 = inverse_dynamics_loss(actions_1, target_actions_1)/1.9
            inverse_dynamics_loss_2 = inverse_dynamics_loss(actions_2, target_actions_2)/1.9

            forward_dynamics_distance = 5 #1 if epoch <= 1 else np.random.randint(1, min(1, max(epoch, 4)))
            forward_dynamics_actions1 = target_actions_1
            forward_dynamics_actions2 = target_actions_2
            forward_dynamics_onehot_actions_1 = torch.zeros((num_frames_i-1, reward_network.ACTION_DIMS), dtype=torch.float32, device=device)
            forward_dynamics_onehot_actions_2 = torch.zeros((num_frames_j-1, reward_network.ACTION_DIMS), dtype=torch.float32, device=device)

            forward_dynamics_onehot_actions_1.scatter_(1, forward_dynamics_actions1.unsqueeze(1), 1.0)
            forward_dynamics_onehot_actions_2.scatter_(1, forward_dynamics_actions2.unsqueeze(1), 1.0)

            forward_dynamics_1 = reward_network.forward_dynamics(mu1[:-forward_dynamics_distance], forward_dynamics_onehot_actions_1[:(num_frames_i-forward_dynamics_distance)])
            forward_dynamics_2 = reward_network.forward_dynamics(mu2[:-forward_dynamics_distance], forward_dynamics_onehot_actions_2[:(num_frames_j-forward_dynamics_distance)])
            for fd_i in range(forward_dynamics_distance-1):
                forward_dynamics_1 = reward_network.forward_dynamics(forward_dynamics_1, forward_dynamics_onehot_actions_1[fd_i+1:(num_frames_i-forward_dynamics_distance+fd_i+1)])
                forward_dynamics_2 = reward_network.forward_dynamics(forward_dynamics_2, forward_dynamics_onehot_actions_2[fd_i+1:(num_frames_j-forward_dynamics_distance+fd_i+1)])

            forward_dynamics_loss_1 = 100 * forward_dynamics_loss(forward_dynamics_1, mu1[forward_dynamics_distance:])
            forward_dynamics_loss_2 = 100 * forward_dynamics_loss(forward_dynamics_2, mu2[forward_dynamics_distance:])

            dt_loss_i = 4*temporal_difference_loss(est_dt_i, torch.tensor(((real_dt_i,),), dtype=torch.float32, device=device))
            dt_loss_j = 4*temporal_difference_loss(est_dt_j, torch.tensor(((real_dt_j,),), dtype=torch.float32, device=device))

            trex_loss = loss_criterion(outputs, labels.long())

            #loss = trex_loss + l1_reg * abs_rewards + reconstruction_loss_1 + reconstruction_loss_2 + dt_loss_i + dt_loss_j + inverse_dynamics_loss_1 + inverse_dynamics_loss_2
            #reconstruction_loss_1 + reconstruction_loss_2 +
            if loss_fn == "trex": #only use trex loss
                loss = trex_loss
            elif loss_fn == "ss": #only use self-supervised loss
                loss = dt_loss_i + dt_loss_j + (inverse_dynamics_loss_1 + inverse_dynamics_loss_2) + forward_dynamics_loss_1 + forward_dynamics_loss_2 + reconstruction_loss_1 + reconstruction_loss_2
            elif loss_fn == "trex+ss":
                loss = dt_loss_i + dt_loss_j + (inverse_dynamics_loss_1 + inverse_dynamics_loss_2) + forward_dynamics_loss_1 + forward_dynamics_loss_2 + reconstruction_loss_1 + reconstruction_loss_2 + trex_loss

            if i < len(training_labels) * validation_split:
                loss.backward()
                optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            # print("total", item_loss)
            cum_loss += item_loss
            if (i + 1) % 1000 == 0:
                #print(i)
                print("loss {}".format(cum_loss))
                print(f'abs rewards: {abs_rewards.item()}')
                cum_loss = 0.0
                # print("check pointing")
                folder = './model_checkpoints'
                torch.save(reward_network.state_dict(), f'{folder}/epoch_{i}.pt')
    print("finished training")

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))

def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return, z1, z2, _, _, _, _ = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)

if __name__ == '__main__':
    env_name = 'breakout'
    loss_fn = 'trex+ss'
    lr = 0.0001
    weight_decay = 0.001
    encoding_dims = 64
    ACTION_DIMS = 4
    reward_model_path = ''
    l1_reg=0.0
    num_iter = 2 if env_name == 'enduro' and loss_fn == 'trex+ss' else 1

    folder = '../training_data'
    training_obs = np.load(f'{folder}/training_obs.npy', allow_pickle=True)
    training_labels = np.load(f'{folder}/training_labels.npy', allow_pickle=True)
    training_actions = np.load(f'{folder}/training_actions.npy', allow_pickle=True)
    training_times = np.load(f'{folder}/training_times.npy', allow_pickle=True)
    demonstrations = np.load(f'{folder}/demonstrations.npy', allow_pickle=True)
    sorted_returns = np.load(f'{folder}/training_obs.npy', allow_pickle=True)

    for i in [training_obs, training_labels, training_actions, training_times, demonstrations, sorted_returns]:
        print(i.shape)


    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(encoding_dims, ACTION_DIMS)
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, training_times, training_actions, num_iter, l1_reg, reward_model_path, loss_fn)
    #save reward network
    torch.save(reward_net.state_dict(), reward_model_path)

    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj[0]) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))