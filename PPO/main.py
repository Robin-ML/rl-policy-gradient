import numpy as np
import time
from os.path import exists
from argparse import ArgumentParser
import torch  
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import gym
from network import Network
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='Run an algorithm on the environment')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')



args = parser.parse_args()
model_path = 'model/model.h5'

def create_env():
    game_name = 'LunarLander-v2'
    env = gym.make(game_name)
    return env

env = create_env()
inp = env.observation_space.shape[0]
output = env.action_space.n

network = Network(inp, output).to(device)


if exists(model_path) == True:
    checkpoint = torch.load(model_path)
    network.load_state_dict(checkpoint)  
    #training_steps =  checkpoint['training_step']   



class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


def get_action(state, action_probs, memory):
        dist = Categorical(action_probs)
        action = dist.sample()
        state = torch.from_numpy(state)       
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

def update(memory, next_state, gamma, epochs, eps_clip, loss_steps, optimizer, network):   
        # TD estimate of state rewards:
        rewards = []
        _, value = network.forward(next_state)
        discounted_reward = value.detach()
        
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # convert list to tensor
        old_states = torch.stack(memory.states)
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for epoch in range(epochs):
            # Evaluating old actions and values :
            policy, value = network.forward(old_states)
            value = torch.squeeze(value)
            dist = Categorical(policy)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)
                
            # Finding Surrogate Loss:
            advantages = rewards - value.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

            mse = nn.MSELoss()
            actor_loss = torch.min(surr1, surr2)
            critic_loss = 0.5 * mse(value, rewards)
            entropy = 0.01 * dist_entropy
            loss = -actor_loss + critic_loss - entropy
                      
            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()  
            
            
            # Log the loss
        writer.add_scalar("1. Total Loss", loss.mean(), loss_steps)
        writer.add_scalar("2. Policy Loss", actor_loss.mean(), loss_steps)
        writer.add_scalar("3. Value Loss", critic_loss.mean(), loss_steps)

def train():
    
    max_episodes = 500
    gamma = 0.95
    learning_rate = 3e-3
    betas = (0.9, 0.999)
    eps_clip = 0.2
    epochs = 5
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=betas)
    
    # logging variables
    max_timesteps = 300
    update_timestep = 5      # update policy every n timesteps
    log_interval = 20
    running_reward = 0
    avg_length, timestep = 0, 0

    mem = Memory()
    start = time.time()
    loss_steps =0

    
    for episode in range(1, max_episodes + 1):
        state = env.reset()

        for t in range(max_timesteps):
            policy, _ = network.forward(state)
            action = get_action(state, policy, mem)
            
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminal:
            mem.rewards.append(reward)
            mem.is_terminals.append(done) 
            
            timestep +=1
            # update if its time
            if timestep % update_timestep == 0:
                loss_steps += 1
                update(mem, state, gamma, epochs, eps_clip, loss_steps, optimizer, network)
                mem.clear_memory()
                writer.flush()
        
            running_reward += reward
            if done:
                break    
        
        avg_length += t

        if episode % 10== 0:
            #Save Model
            torch.save(network.state_dict(), model_path)   

        if episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int(running_reward/log_interval)

            print("{}. Time: {}, Avg Length: {}, Avg Rewards: {}".format(episode, int(time.time()-start) , avg_length, running_reward))

            running_reward = 0
            avg_length = 0
            start = time.time()
            writer.flush()
      
                    

def play():

    with torch.no_grad():
    
        play_ep = 20
        max_timesteps = 300
        try:
            avg = 0
            for e in range(play_ep):
                state = env.reset()
                reward_sum = 0.0

                for _ in range(max_timesteps):
                    #env.render(mode='rgb_array')
                    policy, _ = network.forward(state)
                    dist = policy.cpu().detach().numpy() 
                    action = np.argmax(dist)

                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    if done:
                        break
                
                avg += reward_sum   

                print("{}. Reward: {}".format(e+1, int(reward_sum)))

            print(f"NGB has played {play_ep} Episodes and the average Reward is {int(avg/play_ep)}.") 

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
  