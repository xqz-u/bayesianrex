from train_net import MLP
import torch.nn as nn
import torch

if __name__ == '__main__':
    # Load pretrained model
    hidden_dim = 64
    pretrained = torch.load('../reward_net.params')
    reward_net = MLP(hidden_dim, 5)
    reward_net.load_state_dict(pretrained)

    # Re-initialize last layer
    reward_net.fc2 = nn.Linear(hidden_dim, 1)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    reward_net.to(device)

    # Freeze all parameters
    for param in reward_net.parameters():
        param.requires_grad = False
    
    ## do mcmc :) 
    ## we need some unseen data for this, so either generate more 
    ## or split the data into like 80% train and 20% mcmc?

