# Tutorial Review: Reinforcement Q-Learning from Scratch in Python with OpenAI Gym

## Authors:
* [Satwik Kansal](https://www.learndatasci.com/author/satwik)
* [Brendan Martin](https://www.learndatasci.com/author/brendan)

## Tutorial Link:
<https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/>


## Notes:

### Specify the constraints of the problem:
1. Define the state space for the system:
  * *state space*: set of all possible states that could exist
  * Could be very large and complex
    * this is why OpenAI GYM is used, allows one to develop simulation model of the system
  * __in the tutorial__:
    * 25 possible taxi locations (5x5 grid)
    * 4 possible pick-up/drop-off locations
    * 5 possible passenger locations
    * __total__ = 25 x 4 x 5 = __500__
2. Define the possible actions
  * *actions*: the set of actions the agent can take in any state
  * __in the tutorial__:
    * ```python
      actions = {"south", "north", "east", "west", "pick-up", "drop-off"}
      ```
3. Define the action space
  * *action space*: the set of all the actions that our agent can take in a specific state that is defined within *(is an element of)* the state space
  * __in the tutorial__:
    * The action space for the following state is:
      ```python
      action_space = {"south", "north", "east"}
      ```
      ![state image](https://storage.googleapis.com/lds-media/images/Reinforcement_Learning_Taxi_Env.width-1200.png)
4. Define the rewards/penalties (negative rewards) for the system
  * *rewards*: *" we need to decide the rewards and/or penalties and their magnitude accordingly."*
  * __in the tutorial__:
    * for the state above, we may set the rewards like:
    ```python
    rewards = {"south" : -1, "north" : 1, "east" : 0.5}
    ```

