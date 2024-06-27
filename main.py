from padm_env import ChidEnv
from Q_learning import train_q_learning, visualize_q_table

# User definitions
train = True
visualize_results = True

learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
no_episodes = 1000

goal_coordinates = (4, 4)
hell_state_coordinates = [(3, 2), (0, 4)]

if train:
    env = ChidEnv(goal_coordinates=goal_coordinates,
                  hell_state_coordinates=hell_state_coordinates)

    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                      goal_coordinates=goal_coordinates,
                      q_values_path="q_table.npy")
