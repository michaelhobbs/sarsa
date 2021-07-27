# %% imports and setup
import matplotlib.pyplot as plt
import numpy as np
from env import Grid
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
MAX_EPISODE_COUNT = 1000
LEARNING_RATE = 0.00025

# Use the environment
env = Grid()


# %% init model for Q-Network
num_actions = 8


def create_q_model():
    '''
    Play with different network designs. eg:
    layer1 = layers.Dense(16, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="linear")(layer1)
    '''
    inputs = layers.Input(shape=(2,))

    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(128, activation="relu")(layer1)
    action = layers.Dense(num_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

# %% train
# Adam(learning_rate=0.00025, clipnorm=1.0)
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output (before starting to train, while we are filling buffer)
epsilon_random_frames = 5000  # 50000
# Number of frames for exploration
epsilon_greedy_frames = 100000.0  # 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 20000  # 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 2000  # 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(
                range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array(
                [state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward >= 0.6 or episode_count == MAX_EPISODE_COUNT:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

# %% run single episode with training disabled
test_env = Grid()
step_count = 1
episode_reward = 0
action_history = []
state_history = []
rewards_history = []
done = False
while True:
    state = test_env.state
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    # Take best action
    action = tf.argmax(action_probs[0]).numpy()

    # Apply the sampled action in our environment
    state_next, reward, done = test_env.step(action)
    state_next = np.array(state_next)

    episode_reward += reward

    # Save actions and states in replay buffer
    action_history.append(action)
    state_history.append(state)
    rewards_history.append(reward)
    state = state_next
    step_count += 1

    if done or step_count > 10000:
        break

# %%
model.save('above_0.6.h5')

# %% quiver plot
import math

def cellCoordsByIndex(index, xAxisSteps, yAxisSteps, xDelta, yDelta):
    '''
    Converts an index of flattened 2d space to [x, y] coordinates.
    The first cell has index 0 and coordinates [0, 0].
    The last cell has index _numCells_ and coordinates [1, 1].

    args: \n
            index : number in [0, (numCells - 1)]
            xAxisSteps : number of columns
            yAxisSteps : number of rows
            xDelta : distance between 2 cells along the x-axis
            yDelta : distance between 2 cells along the y-axis

    returns: [x, y] coordinate tuple for the centre of the i-th place cell
    '''
    return np.array([(index % xAxisSteps) * xDelta, (math.floor(index / xAxisSteps) * yDelta)])


def generateCoordsArray(xAxisSteps, yAxisSteps):
    numPlaceCells = xAxisSteps * yAxisSteps
    xDelta = 1 / (xAxisSteps - 1)
    yDelta = 1 / (yAxisSteps - 1)
    return np.array([cellCoordsByIndex(j, xAxisSteps, yAxisSteps, xDelta, yDelta) for j in range(numPlaceCells)])


def max_action(state):
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    action = tf.argmax(action_probs[0]).numpy()
    return action


axisSize = 20
grid_cells = generateCoordsArray(axisSize, axisSize)
max_actions = list(map(max_action, grid_cells))

xDelta = 1 / (axisSize - 1)
axis = [(j % axisSize) * xDelta for j in range(axisSize)]

x = axis
y = axis

X, Y = np.meshgrid(x, y)

maxIndexes = max_actions


def convertIndexToX(i):
    return np.array([0, 0.7, 1, 0.7, 0, -0.7, -1, -0.7])[i]


def convertIndexToY(i):
    return np.array([-1, -0.7, 0, 0.7, 1, 0.7, 0, -0.7])[i]


u = [convertIndexToX(maxIndexes[:])]
v = [convertIndexToY(maxIndexes[:])]
_, ax = plt.subplots(figsize=(9, 9))
rectangle = plt.Rectangle((0, 0), 1, 1, ec='blue', fc='none')
plt.gca().add_patch(rectangle)
circle2 = plt.Circle((0.8, 0.8), 0.1, color='r', fill=False)
ax.add_artist(circle2)
plt.plot([0.1], [0.1], 'rx')
ax.quiver(X, Y, u, v)
# ax.quiver(X,Y,u,v, scale=2*axisSize, scale_units='xy')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.axis([-0.1, 1.1, -0.1, 1.1])
ax.set_aspect('equal')
plt.show()
# plt.close()

# %%
