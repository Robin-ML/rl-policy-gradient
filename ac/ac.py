import numpy as np
import time
import os
import argparse
import torch  
import torch.optim as optim

import gym
from network import Network


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Run an algorithm on the environment')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')



args = parser.parse_args()
model_path = 'model/acTD_model.h5'



def create_env():
    game_name = 'LunarLander-v2'
    env = gym.make(game_name)
    return env

env = create_env()
inp = env.observation_space.shape[0]
output = env.action_space.n

network = Network(inp, output).to(device)


if os.path.exists(model_path) == True:
    network.load_state_dict(torch.load(model_path))     

learning_rate = 3e-4
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

class Memory:
    def __init__(self):
        self.values = []
        self.logprobs = []
        self.rewards = []
        

    def clear_memory(self):
        del self.values[:]
        del self.rewards[:]
        del self.logprobs[:]


def update(done, next_state, memory, gamma, entropy_term, optimizer, loss_steps, network):   
        # TD estimate of state rewards:
       if done:
            Qval = 0
       else:
          _, Qval= network.forward(next_state)
          Qval = Qval.cpu().detach().numpy()                
       
        # compute Q values
       Qvals = []

       for t in reversed(range(len(memory.rewards))):
            Qval = memory.rewards[t] + gamma * Qval
            Qvals.append(Qval)
        
       Qvals.reverse()
        

        #update actor critic
       values = torch.Tensor(memory.values).to(device)
       Qvals = torch.FloatTensor(Qvals).to(device)
       Qvals = (Qvals - Qvals.mean()) / (Qvals.std() + 1e-5)

       log_probs = torch.stack(memory.logprobs).to(device)
        
       advantage = Qvals - values
       actor_loss = (-log_probs * advantage).mean()
       critic_loss = 0.5 * advantage.pow(2).mean()
       loss = actor_loss + critic_loss + 0.001 * entropy_term

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       return loss


def train():
    max_episodes = 100
    gamma = 0.99
    update_steps = 5
    for episode in range(max_episodes):
        start = time.time()
        state = env.reset()
        mem = Memory()
        reward_sum = 0
        done =  False
        entropy_term = 0
        steps =0

        while not done:
            policy, value = network.forward(state)
            value = value.cpu().detach().numpy() 

            mem.values.append(value)
            dist = policy.cpu().detach().numpy() 
            dist = dist[dist != 0]
            action = np.random.choice(output, p=np.squeeze(dist))
            
            log_prob = torch.log(policy.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            
            next_state, reward, done, _ = env.step(action) 

            mem.rewards.append(reward)
            mem.logprobs.append(log_prob)
            entropy_term += entropy
            reward_sum += reward
            state = next_state
            steps +=1
            
            if done or steps % update_steps ==0 :
                loss = update(done, next_state, mem, gamma, entropy_term, optimizer, steps, network)
                mem.clear_memory()
        
        if episode % 50 == 0:
            #Save Model
            torch.save(network.state_dict(), model_path)
            print("{}. Time: {}, Reward: {}, Loss: {}".format(episode+1, time.time()-start ,reward_sum, loss))
        
            
                    


def play():
        play_ep = 1
        try:
            avg = 0
            for _ in range(play_ep):
                state = env.reset()
                reward_sum = 0.0
                done = False
                    
                while not done:
                    policy, _ = network.forward(state)
                    dist = policy.cpu().detach().numpy() 
                    action = np.argmax(dist)

                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                
                avg += reward_sum   

            print(f"NGB has played {play_ep} Episodes and the average Reward is {avg/play_ep} .") 

        except KeyboardInterrupt:
                print("Received Keyboard Interrupt. Shutting down.")
        finally:
                env.close()
    
            


if __name__ == '__main__':
  print(args)
  if args.train:
      train()
  
  else:
      play()
  