import numpy as np
import torch  
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    ''' Policy and value prediction from inner abstract state '''
    def __init__(self, state, actions, hidden_size=56):
        super().__init__()
        self.p1 = nn.Linear(state, hidden_size)
        self.p2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, actions) 

        self.v1 = nn.Linear(state, hidden_size)
        self.v2 = nn.Linear(hidden_size, hidden_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.Tensor(x).to(device)

        p = F.relu(self.p1(x))
        p  = F.relu(self.p2(p))
        policy = F.softmax(self.actor(p), dim=-1)
        
        v  = F.relu(self.v1(x))
        v  = F.relu(self.v2(v))
        value = self.critic(v)
        
        return policy, value



