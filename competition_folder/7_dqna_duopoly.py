import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.utils import common
from tf_agents.trajectories import Trajectory, PolicyStep, time_step as ts
from keras.layers import Dense
from environment_functions import DynamicPricingCompetition, AirlineEnvironment, ExternalSimulator

import datetime

start = datetime.datetime.now()

# Settings
max_action = 100
min_action = 20
action_price_step = 2
num_actions = (max_action - min_action) / action_price_step
num_features = 33

discount = 1.0
comp_sellout_price = 100
early_termination_penalty = 10
price_diff_penalty = 0.1
loadfactor_diff_penalty = 0.4
stock_remainder_penalty = 50

hidden_units_layer1 = 20
hidden_units_layer2 = 40
learning_rate = 1e-3
beta_1 = 0.9
beta_2 = 0.999
target_update_period = 2000

policy = 'boltzmann_temperature'
exponential_decay_rate = 1.0
boltzmann_temperature_start = 100.0
boltzmann_temperature_end = 1.0
epsilon_greedy_start = 1.0
epsilon_greedy_end = 0.01
decay_steps = 5 * 1000 * 100

replay_buffer_batch_size = 1
replay_buffer_max_size = 10000
sample_batch_size = 64
num_steps = 2
num_parallel_calls = 4

seed = 123
tf.random.set_seed(seed)

# Environments
dpc_game = DynamicPricingCompetition()
simulator = ExternalSimulator()
environment = AirlineEnvironment(dpc_game, simulator, num_features, num_actions, discount, min_action,
                                 action_price_step, comp_sellout_price, early_termination_penalty,
                                 price_diff_penalty, loadfactor_diff_penalty, stock_remainder_penalty)
train_env = tf_py_environment.TFPyEnvironment(environment)

# Set up the agent and the network for the agent
init = tf.keras.initializers.HeUniform()
layer1 = Dense(units=hidden_units_layer1, input_shape=(num_features,), activation='relu',
               kernel_initializer=init, name='hidden_layer1')
layer2 = Dense(units=hidden_units_layer2, activation='relu', kernel_initializer=init, name='hidden_layer2')
layer3 = Dense(units=num_actions, activation=None, kernel_initializer=init)
q_net = sequential.Sequential([layer1, layer2, layer3])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

train_step_counter = tf.Variable(0)

if policy == 'boltzmann_temperature':
    if exponential_decay_rate == 1:
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=boltzmann_temperature_start,
            decay_steps=decay_steps,
            end_learning_rate=boltzmann_temperature_end)
    else:
        epsilon_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=boltzmann_temperature_start,
            decay_steps=decay_steps,
            decay_rate=exponential_decay_rate,
        )

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=discount,
        boltzmann_temperature=lambda: epsilon_fn(train_step_counter),
        epsilon_greedy=None,
        train_step_counter=train_step_counter
    )
elif policy == 'epsilon_greedy':
    if exponential_decay_rate == 1:
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=epsilon_greedy_start,
            decay_steps=decay_steps,
            end_learning_rate=epsilon_greedy_end)
    else:
        epsilon_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=epsilon_greedy_start,
            decay_steps=decay_steps,
            decay_rate=exponential_decay_rate,
        )

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=discount,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter),
        train_step_counter=train_step_counter
    )

agent.initialize()

# replay buffer and driver for training
replay_buffer = TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=replay_buffer_batch_size,
    max_length=replay_buffer_max_size
)

# Make reusing a model possible
train_checkpointer = common.Checkpointer(
    ckpt_dir='./participant_folder',
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)

# Setup training
dataset = replay_buffer.as_dataset(
    sample_batch_size=sample_batch_size,
    num_steps=num_steps,
    num_parallel_calls=num_parallel_calls
).prefetch(num_parallel_calls)
iterator = iter(dataset)

agent.train = common.function(agent.train)


def p(
        current_selling_season,
        selling_period_in_current_season,
        prices_historical_in_current_season=None,
        demand_historical_in_current_season=None,
        competitor_has_capacity_current_period_in_current_season=True,
        information_dump=None,
):
    """
    Generates a linear increase as a baseline strategy to test the waters.

    input:
        current_selling_season:
                int of current selling season 1..100
        selling_period_in_current_season:
                int of current period in current selling season 1..100
        prices_historical_in_current_season:
                numpy 2-dim array: (number competitors) x (past iterations)
                it contains the past prices of each competitor
                (you are at index 0) over the past iterations
        demand_historical_in_current_season:
                numpy 1-dim array: (past iterations)
                it contains the history of your own past observed demand
                over the last iterations
        competitor_has_capacity_current_period_in_current_season:
                boolean indicator if the competitor has some free capacity
                at the beginning of the current period/ selling interval
        information_dump:
                some information object you like to pass to yourself
                at the next iteration
    """
    if (current_selling_season == 1) & (selling_period_in_current_season == 1):
        
        time_step = train_env.reset()

        action_step = agent.collect_policy.action(time_step)
        price = min_action + int(action_step.action) * action_price_step

        # prev_step = time_step
        # prev_action_step = action_step

        step_type = np.array(time_step[0])
        reward = np.array(time_step[1])
        discount = np.array(time_step[2])
        observation = np.array(time_step[3])

        information_dump = {
            # "iterator": iterator,
            # "prev_step": prev_step,
            # "prev_action_step": prev_action_step,
            "prev_price": price,
            "prev_discount": discount,
            "prev_observation": observation,
            "prev_reward": reward,
            "prev_step_type": step_type,
            "soldout": False
        }

        return (price, information_dump)

    else:

        if selling_period_in_current_season == 1:

            # TODO: No (online) feedback from last period, so will have to predict seats sold
            # and last competitor price in last period (if time period was 100)
            if (dpc_game.selling_period == 100) & (dpc_game.loadfactor < 80):
                last_demand = 1
            else:
                last_demand = dpc_game.demand_t1

            last_price = dpc_game.price_t1
            last_comp_price = dpc_game.price_competitor_t1
            competitor_has_capacity = dpc_game.competitor_has_capacity

            prev_step = ts.TimeStep(
                discount=tf.constant(information_dump['prev_discount']),
                observation=tf.constant(information_dump['prev_observation']),
                reward=tf.constant(information_dump['prev_reward']),
                step_type=tf.constant(information_dump['prev_step_type'])
            )
            prev_action_step = PolicyStep(
                action=(
                    tf.constant([np.max([0, int((information_dump['prev_price'] - min_action) / action_price_step)])])),
                state=(), info=()
            )

            simulator.update(last_price, last_comp_price, last_demand, competitor_has_capacity)
            time_step = train_env.step(prev_action_step)

            # Package information into a trajectory
            traj = Trajectory(
                prev_step.step_type,
                prev_step.observation,
                prev_action_step.action,
                prev_action_step.info,
                time_step.step_type,
                time_step.reward,
                time_step.discount
            )
            replay_buffer.add_batch(traj)

            time_step = train_env.step(prev_action_step)
            action_step = agent.collect_policy.action(time_step)
            price = min_action + int(action_step.action) * action_price_step

            # iterator = information_dump['iterator']
            if replay_buffer.num_frames() >= replay_buffer_max_size // 2:
                trajectories, _ = next(iterator)
                _ = agent.train(experience=trajectories)

            step_type = np.array(time_step[0])
            reward = np.array(time_step[1])
            discount = np.array(time_step[2])
            observation = np.array(time_step[3])

            information_dump = {
                # "iterator": iterator,
                # "prev_step": prev_step,
                # "prev_action_step": prev_action_step,
                "prev_price": price,
                "prev_discount": discount,
                "prev_observation": observation,
                "prev_reward": reward,
                "prev_step_type": step_type,
                "soldout": False
            }

            return (price, information_dump)

        else:

            # if (selling_period_in_current_season == 100) & (current_selling_season == 5):
            #     print(f'Time for 5 seasons: {datetime.datetime.now() - start}')

            if information_dump['soldout'] == True:
                # Passing because we sold out
                pass
            else:

                last_demand = demand_historical_in_current_season[-1]
                last_price = prices_historical_in_current_season[0][-1]
                last_comp_price = prices_historical_in_current_season[1][-1]
                competitor_has_capacity = competitor_has_capacity_current_period_in_current_season

                prev_step = ts.TimeStep(
                    discount=tf.constant(information_dump['prev_discount']),
                    observation=tf.constant(information_dump['prev_observation']),
                    reward=tf.constant(information_dump['prev_reward']),
                    step_type=tf.constant(information_dump['prev_step_type'])
                )
                prev_action_step = PolicyStep(
                    action=(
                        tf.constant([np.max([0, int((information_dump['prev_price'] - min_action) / action_price_step)])])),
                    state=(), info=()
                )

                simulator.update(last_price, last_comp_price, last_demand, competitor_has_capacity)
                time_step = train_env.step(prev_action_step)

                action_step = agent.collect_policy.action(time_step)
                price = min_action + int(action_step.action) * action_price_step

                # Add trajectory and adding to replay buffer + training
                # prev_step = information_dump['prev_step']
                # prev_action_step = information_dump['prev_action_step']

                # Package information into a trajectory
                traj = Trajectory(
                    prev_step.step_type,
                    prev_step.observation,
                    prev_action_step.action,
                    prev_action_step.info,
                    time_step.step_type,
                    time_step.reward,
                    time_step.discount
                )
                replay_buffer.add_batch(traj)

                # iterator = information_dump['iterator']
                if int(replay_buffer.num_frames()) >= (replay_buffer_max_size // 2):
                    trajectories, _ = next(iterator)
                    _ = agent.train(experience=trajectories)

                step_type = np.array(time_step[0])
                reward = np.array(time_step[1])
                discount = np.array(time_step[2])
                observation = np.array(time_step[3])

                # print(f'last demand: {last_demand}, current stock {simulator._our_stock}')

                information_dump = {
                    # "iterator": iterator,
                    # "prev_step": prev_step,
                    # "prev_action_step": prev_action_step,
                    "prev_price": price,
                    "prev_discount": discount,
                    "prev_observation": observation,
                    "prev_reward": reward,
                    "prev_step_type": step_type,
                    "soldout": True if simulator._our_stock <= 0 else False,
                    "last_demand": last_demand
                }

            return (price, information_dump)
