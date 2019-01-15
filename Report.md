**Udacity's Deep Reinforcement Learning Nanodegree**

# Report on Navigation Project 





##Learning Algorithm



```pyth
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_score=15):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        t0 = time.time()
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state, reward, done = extract_info(env_info)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        t1 = time.time() - t0
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tAverage Score: {:.2f} in {:.2f} sec'.format(i_episode, np.mean(scores_window), t1), end="")
    
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

#         if np.mean(scores_window)>=target_score or (i_episode % 20 == 0 and not is_making_progress(scores_window)):
        if np.mean(scores_window)>=target_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tMin Score: {:.2f}'.format(i_episode-100, np.mean(scores_window), np.min(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'model.pt')
            break
            
    return scores
```





##Network Architecture for Deep Q-Network (DQN)



```py
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=32, fc3_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.dropout = nn.Dropout(0)

    def forward(self, state):
        x = self.dropout(F.relu(self.fc1(state)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.fc4(x)
```





##Experiments







##Plot of Rewards







##Ideas for Future Work