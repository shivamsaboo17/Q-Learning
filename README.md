# Q-Learning
Q-Learning algorithm in numpy

## Usage:
```python
# import the files
from q_learn import QLearn
# Create model
model = QLearn(gym_env_name, total_episodes, learning_rate, max_steps, discount_factor)
# Training
model.fit()
#Test the model for number of episodes
model.test(test_episodes=100)
# Save the model
model.save(model_name)
# Load model state dict
model.load_state_dict(model_name)
# Plot rewards
model.plot()
# Plot scheduled epsilon 
model.plot(epsilon=True)
```
## Requirements:
1. Numpy
2. Matplotlib
3. tqdm
4. OpenAI Gym
5. Pickle 

## Working
Q-Learning works by building a table of action and states and learning the 'q-values' at each location. The Q-Table looks something like this</br>
![](https://cdn-images-1.medium.com/max/1200/1*ut7-8VVa-TWC40_YAeqZ7Q.png)</br></br>
The values in the table are randomly (or zero) initialized at first and are learnt through the bellman equation.</br>
The bellman equation is</br></br>
![](https://cdn-images-1.medium.com/max/1500/1*jmcVWHHbzCxDc-irBy9JTw.png)</br></br>
The learning starts by first focussing more on the exploration and then on the exploitation. This is because in the start we don't know which is the optimal step and hence we must try to explore entire space and later do exploitation. Exploration simply means randomly selecting an action and exploitation means selecting the action with highest reward. This transition can be easily modelled using exponentially or linearly scheduling the exploration probability.</br></br>
![](https://cdn-images-1.medium.com/max/1200/1*9StLEbor62FUDSoRwxyJrg.png)</br></br>
This learnt q-table can then be used to play simple gym games like Taxi-v2 or FrozenLake-v0!
