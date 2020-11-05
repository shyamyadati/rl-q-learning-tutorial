"""
Description:
    tutorial python script - implements all the code from the tutorial in a single python script

NOTE:
    for more information about the environment see: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
"""

# =============================================================================
# standard library imports
import time
import random

# -------------------------------------
# third party imports
import gym  # OpenAI Gym
import numpy as np

# -------------------------------------
# local imports


# =============================================================================
# helper functions
def header(message):
    """
    Description:
        prints out a header in std out
    """
    print(f"\n{'-'*(len(message)+2)}\n {message}\n{'-'*(len(message)+2)}\n")

# -------------------------------------
def print_locations(env):
    """
    Description
        Print out the location of where the taxi, passenger, and destination are
    """
    # the following definitions were taken from https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
    passenger_locations = ["R", "G", "Y", "B", "In Taxi"]
    destinations = ["R", "G", "Y", "B"]

    # get the positions for where the taxi, passenger and destination progromatically
    taxi_row, taxi_col, pass_idx, dest_idx = env.decode(env.s)
    print(f"Current State: {env.s}")
    print(f"    taxi (row, col) : ({taxi_row}, {taxi_col})")
    print(f"    passenger       : {passenger_locations[pass_idx]}")
    print(f"    destination     : {destinations[dest_idx]}")


# -------------------------------------
def print_rewards_table(env):
    """
    Description:
        Prints the reward table of the current state
    """

    # the following definition was taken from https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
    action_space_map = ["move south", "move north", "move east", "move west", "pickup passenger", "dropoff passenger"]

    # get the current reward table
    reward_table = env.P[env.s]
    
    # each row in the table for this environment has the following format
    # "action number" : [(probablility, next state, reward, done)]
    print(f"Rewards Table for state: {env.s}")
    for action_number, v in reward_table.items():
        action_str = action_space_map[action_number]
        for prob, nxt_state, reward, done in v:
            print(f"    {action_str}:")
            print(f"        probability : {prob}")
            print(f"        next state  : {nxt_state}")
            print(f"        reward      : {reward}")
            print(f"        done        : {done}")

# -------------------------------------
def replay(env, frames, time_step=0.01):
    """
    replays a simulation given the env, and frames captured during the simulation
    """
    print("Replaying simulation:\n")
    env.s = 328
    env.render()
    time.sleep(1)
    for f in frames:
        for i in range(8):
            print ("\033[A\033[A") # clear one line from stdout
        env.s = f['state']
        env.render()
        time.sleep(time_step)

# -------------------------------------
def train_q_learning_agent(env, num_epochs=100000):
    """
    Description:
        train_agent will train the Q-learning agent 
    Returns:
        the q_table for the agent
    """
    # create the q_table size of the observation_space (state space) x action_space
    print("Training the Q-learning agent")
    print(f"Creating q_table of ({env.observation_space.n} x {env.action_space.n})")
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # set up the hyperparameters for the Q-learning model
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # setup training variables
    all_epochs = []
    all_penalties = []

    print("Staring training...")
    start_time = time.time()
    for i in range(num_epochs):
        # at the beginning of each epoch - start with a new random state
        state = env.reset()

        # reset the epoch's loop variables
        epochs, penalties, reward, = 0, 0, 0
        done = False
    
        while not done:
            # get an action - either a random one depending on epsilon, or 
            # one from the learned values
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            # apply the selected action to the environment
            next_state, reward, done, info = env.step(action) 
        
            # get the old value from the table
            old_value = q_table[state, action]   

            # get the max predicted award given the next state
            next_max = np.max(q_table[next_state])   

            # Update the table
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
        if i%100 == 0:
            if i != 0:
                print ("\033[A\033[A") # clear one line from stdout
            print(f"    Episode: {i}")

    print ("\033[A\033[A") # clear one line from stdout
    print(f"    Episode: {num_epochs}")
    print(f"Training finished - elapsed time: {(time.time() - start_time):0.3f} s")
    return q_table

# -------------------------------------
def run_sim(env, q_table=None):
    """
    Description:
        Runs a random simulation given the environment
    """
    env.s = 328  # set environment to illustration's state

    # initalize the simlation variables
    epochs = 0
    penalties, reward = 0, 0
    frames = [] # for animation
    done = False

    # run through the simulation
    print("Running the simulation ...")
    while not done:
        if q_table is None: 
            # select a random action
            action = env.action_space.sample() 
        else:
            #else, if the agent was provided us it
            action = np.argmax(q_table[env.s])
           
        # Apply the selected action to the environment
        state, reward, done, info = env.step(action)

        # count the number of penalties that we incurr
        if reward == -10:
            penalties += 1
        
        # Put each rendered frame into dict for animation after
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

        # increment the epoc count
        epochs += 1
    
    return epochs, penalties, frames

# =============================================================================
# start main
if __name__ == "__main__":
    # -------------------------------------
    # create the taxi environment
    # NOTE: The tutorial states v2, but this caused a:
    # gym.error.DeprecatedEnv: Env Taxi-v2 not found (valid versions include ['Taxi-v3'])
    header("Creating the environemnt")
    env = gym.make("Taxi-v3").env

    # Render the environment to visualize it
    # NOTE: This will create the environment in a random state
    print("NOTE:")
    print("    | is a wall")
    print("    'R','G','Y','B' are pick-up/drop-off locations")
    print("    BLUE:    letter is passenger")
    print("    MAGENTA: destination")
    print("    YELLOW filled cursor is current location of empty taxi")
    print("    GREEN filled cursor is current location of full taxi")
    print(f"Current state number: {env.s}")

    env.render()
    print_locations(env)
    print_rewards_table(env)
    input("\nPress enter to continue")

    # ------------------------------------
    # If we want to reset to a new random state:
    header("Resetting the environment to a new random state")
    env.reset()
    env.render()
    print_locations(env)
    input("\nPress enter to continue")

    # ------------------------------------
    # IF we want to set a specific state:
    header("Setting to state: 328 -> (3, 1, 2, 0)")
    state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
    env.s = state
    env.render()
    print_locations(env)
    input("\nPress enter to continue")

    # ---------------------------------
    # solve without using RL
    header("Try and solve the problem without reinforcement learning")
    epochs, penalties, frames = run_sim(env)
    print(f"Timesteps taken    : {epochs}")
    print(f"Penalties incurred : {penalties}")
    choice = input("Replay simulation (y/n)? ")
    if choice.lower()[0] == 'y':
        replay(env, frames)
        input("\nPress enter to continue")

    # ---------------------------------
    header("Solve the problem using Q-Learning")
    q_table = train_q_learning_agent(env)
    epochs, penalties, frames = run_sim(env, q_table)
    print(f"Timesteps taken    : {epochs}")
    print(f"Penalties incurred : {penalties}")
    choice = input("Replay simulation (y/n)? ")
    if choice.lower()[0] == 'y':
        replay(env, frames, time_step=0.5)
        input("\nPress enter to continue")




