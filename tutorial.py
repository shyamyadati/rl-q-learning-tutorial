"""
Description:
    tutorial python script - implements all the code from the tutorial in a single python script

NOTE:
    for more information about the environment see: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
"""

# =============================================================================
# standard library imports
import time

# -------------------------------------
# third party imports
import gym  # OpenAI Gym

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
def run_random_sim(env):
    """
    Description: Runs a random simulation given the environment
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
        # select a random action
        action = env.action_space.sample() 

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

# -------------------------------------
def replay(env, frames):
    """
    replays a simulation given the env, and frames captured during the simulation
    """
    print("Replaying simulation:\n")
    env.s = 328
    env.render()
    for f in frames:
        time.sleep(0.01)
        for i in range(8):
            print ("\033[A\033[A") # clear one line from stdout
        env.s = f['state']
        env.render()

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
    epochs, penalties, frames = run_random_sim(env)
    print(f"Timesteps taken    : {epochs}")
    print(f"Penalties incurred : {penalties}")
    replay(env, frames)
    input("\nPress enter to continue")



