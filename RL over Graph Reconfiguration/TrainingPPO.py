# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:24:39 2024

@author: UTD
"""


import numpy as np
import tensorflow as tf
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size=64, actor_lr=2e-6, critic_lr=1e-6, gamma=0.99, clip_ratio=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def build_actor(self):
        input_layer = tf.keras.layers.Input(shape=(self.state_dim+5,))
        hidden_layer = tf.keras.layers.Dense(self.hidden_size, activation='relu')(input_layer)
        #mu = tf.keras.layers.Dense(self.action_dim, activation='tanh')(hidden_layer)
        #sigma = tf.keras.layers.Dense(self.action_dim, activation='softplus')(hidden_layer)
        mu = tf.keras.layers.Dense(1, activation='tanh')(hidden_layer)
        #scaled_mu = 127.5 * (mu + 1.0)  # Scale and shift the output to [0, 255]
        #print(scaled_mu)
        sigma = tf.keras.layers.Dense(1, activation='softplus')(hidden_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=[mu, sigma])
        return model

    def build_critic(self):
        #print("Critic Initialization:")

        input_layer = tf.keras.layers.Input(shape=(self.state_dim + self.action_dim + 5,))
        hidden_layer = tf.keras.layers.Dense(self.hidden_size, activation='relu')(input_layer)
        value = tf.keras.layers.Dense(1)(hidden_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=value)
        return model


    def get_action(self, state):
        #print("Action Initialization")
        state = np.expand_dims(state, axis=0)
        mu, sigma = self.actor(state)
        #print(mu)
        #print(sigma)
        random = np.abs(np.random.normal(0, 1, size=sigma.shape))
        action = (mu + random * sigma)*100
        #print(action)
        action = int(action[0][0])  # Convert to integer
        #print("Action Initialized")
        #action = np.clip(action, 0, 255)  # Clip the action to [0, 255]
        
        return action
    
    
    def compute_loss(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            #print("Loss Initialization:")

            advantages = self.compute_advantages(states, rewards, next_states, dones)
            mu, sigma = self.actor(states)
            log_prob = self.gaussian_likelihood(actions, mu, sigma)

            critic_values = tf.squeeze(self.critic(states), axis=-1)

            ratios = tf.exp(log_prob - self.gaussian_likelihood(actions, mu, sigma))
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = (-tf.reduce_mean(tf.minimum(surr1, surr2)))/1000000
            actor_losses.append(actor_loss)
            dones = tf.cast(dones, dtype=tf.float32)
            critic_loss = (tf.reduce_mean(tf.square(rewards + self.gamma * self.critic(next_states) * (1 - dones) - critic_values)))/1000000000000
            critic_losses.append(critic_loss)

            total_loss = actor_loss + critic_loss

        return total_loss

    def compute_advantages(self, states, rewards, next_states, dones):
        #print("Advatages Initialization:")
        #print(next_states)

        next_values = self.critic(next_states)
        #print("Critic Done!")

        dones = tf.cast(dones, dtype=tf.float32)
        #print(dones)
        #print(next_values)
        
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        #print(td_targets)
        #print(states)

        advantages = td_targets - self.critic(states)
        #print(advantages)
        return advantages

    def train(self, states, actions, rewards, next_states, dones, epochs=10, batch_size=64):
        #print("Training Initialization:")
        dones = np.array(dones, dtype=bool)
        #print("Training Initialization2:")
        dataset = tf.data.Dataset.from_tensor_slices((states.astype(np.float32),actions.astype(np.float32),rewards.astype(np.float32),next_states.astype(np.float32),dones.astype(np.bool))).shuffle(buffer_size=1000)
        #print("Training Initialization3:")
        dataset = dataset.batch(batch_size)
        #print("Training Initialized")

        for _ in range(epochs):
            #print("Loop Initalized")

            for batch in dataset:
              
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = batch
                #print("Hello")
                with tf.GradientTape() as tape:
                    #print("Loop3 Initalized")

                    loss = self.compute_loss(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)
                
                total_losses.append(loss)
                grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
                self.actor_opt.apply_gradients(zip(grads[:len(self.actor.trainable_variables)], self.actor.trainable_variables))
                self.critic_opt.apply_gradients(zip(grads[len(self.actor.trainable_variables):], self.critic.trainable_variables))

    def gaussian_likelihood(self, actions, mu, sigma):
        return -0.5 * (tf.math.log(2 * np.pi * sigma ** 2) + (actions - mu) ** 2 / (sigma ** 2))




# Environment

#from gym.envs.registration import register
#from pyinterfaceV3 import ShipEnvironment
# register(
#    id='ShipEnvironment-v2',
#    entry_point='py_interface - V2:ShipEnvironment',  # Update 'your_module_name' with the actual module name
#)

proj_path = r"C:\Users\UTD\Box\New folder (sxs190214@utdallas.edu)\ONR\reconfiguration\Roshni Environment\Two_Zone_MVDC.prj"
SimModel = "Two_Zone_MVDC_PSolver_R2020b" #name of the simulink model


env = ShipEnvironment(proj_path, SimModel)
#env = gym.make('env')  # Replace 'YourShipEnv-v0' with the name of your environment

# Agent
state_dim = 1
#state_dim = env.observation_space.shape[0]

#action_dim = env.action_space.shape[0]
action_dim = 0
agent = PPOAgent(state_dim, action_dim)
# Training
epochs = 200
# for epoch in range(epochs):
#     state = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action = agent.get_action(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.train(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))
#         state = next_state
#         total_reward += reward
#     print("Epoch:", epoch, "Total Reward:", total_reward)


epoch_rewards = []
actor_losses = []
critic_losses = []
total_losses = []



for epoch in range(epochs):
    state = env.reset()
    state = state[:,-1,0]
    
    
    state[2,] = state[2,]/12000
    state[3,] = state[2,]/12000
    state[4,] = state[2,]/1440000
    state[5,] = state[2,]/1440000

    done = False
    counter = 0 
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        
        if action > 255 : 
            action = 255
        elif action < 0 : 
            action = 0
            
        #action = random.randint(0, 255)
        #print("1", action, type(action))
        next_state_new, reward, done, _ = env.step(action)
        #print("2")
        next_state = next_state_new[:,-1,0]
        
        
        next_state[2,] = next_state[2,]/12000
        next_state[3,] = next_state[2,]/12000
        next_state[4,] = next_state[2,]/1440000
        next_state[5,] = next_state[2,]/1440000
        #print("3")
        #done = np.array(False, dtype=bool)

        agent.train(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))
        #print("4")
        state = next_state
        #print("5")
        reward = reward / 1000000
        total_reward += reward
        #print("6")
        print("Epoch:", epoch, "Action:", action, "Reward:", reward, "Done:", done,"Total Reward:", total_reward)
        
        counter = counter + 1 
        if counter == 4 : done = True
    
    epoch_rewards.append(total_reward)

    
    






plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(epoch_rewards)
plt.title('Total Reward per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Total Reward')

plt.subplot(2, 2, 2)
plt.plot(actor_losses)
plt.title('Actor Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Actor Loss')

plt.subplot(2, 2, 3)
plt.plot(critic_losses)
plt.title('Critic Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Critic Loss')

plt.subplot(2, 2, 4)
plt.plot(total_losses)
plt.title('total Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('total Loss')

plt.tight_layout()
plt.show()



env.close()



















