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

