[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

**Udacity's Deep Reinforcement Learning Nanodegree**

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Project Detail

* PyTorch is used as deep learning framework for defining DQN. If we are curious about the theory of DQN used in this project, please refer to [Report](./Report.pdf)
* Unity is used as an RL environment and its detail is as below.
  * One agent is configured with the name of "BananaBrain", and the size of its action spaces is 4.
  * The size of a states is 37 dimension, and it is not fixel based.

```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```

* An episode runs up to 300 steps
* To goal of this project is to acheive an average cumulative score over 100 episode 13 or higher. In this project, I increase the target average rewards to 15 from 13 in order to make the agent play well on a single episode.
* If you are curious about the detail algorithm and the performance with the trained agent, refer to [Report](./Report.md) ([PDF](./Report.pdf) version)


### Codes in this project

- Navigation.ipynb
  - A Jupyter notebook where all the code execution happens from RL environment creation, RL training, and testing.
- dqn_agent.py
  - A module which defines Agent class and ReplayBuffer for experience replay.
  - Agent class;
    - chooses an action using the policy
    - updates the replay buffer, and trigger DQN training
    - executes DQN training and updates the target policy gradually
  - ReplayBuffer class;
    - provides an interface for adding a tuple into the buffer
    - provides a mini-batch from the buffer for mini-batch SGD training
- model.py
  - QNetwork class is defines a DQN model in PyTorch

### Getting Started - Install Unity environment

In order to run this reinforcement learning example, you need to install the environment as well as python and PyTorch. The below is to guide you how to install the environment per your OS environment.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the clone repository folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  Or you can load `model.pt` which I trained already.

### Curious on where to apply RL?

[Asynchronous Advantage Actor-Critic Agent for Starcraft II](https://arxiv.org/abs/1807.08217)



