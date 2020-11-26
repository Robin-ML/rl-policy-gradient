import numpy as np
import torch  
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    ''' Policy and value prediction from inner abstract state '''
    def __init__(self, state, actions, hidden_size=128):
        super().__init__()
        self.linear1 = nn.Linear(state, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.l0 = nn.Linear(hidden_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)


        self.actor = nn.Linear(hidden_size, actions) 

        self.linear4 = nn.Linear(state, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        p = F.relu(self.linear1(x))
        p1  = F.relu(self.linear2(p))
        p2  = F.relu(self.linear3(p))
        p3  = F.relu(self.l0(p))
        p4  = F.relu(self.l1(p))
        p = p1 + p2 + p3 + p4
        policy = F.softmax(self.actor(p), dim=-1)
        
        v  = F.relu(self.linear4(x))
        v1  = F.relu(self.linear5(v))
        v2  = F.relu(self.linear6(v))
        v3  = F.relu(self.l2(v))
        v4  = F.relu(self.l3(v))
        v = v1 + v2 +v3 + v4
        value = self.critic(v)
        
        return policy, value



