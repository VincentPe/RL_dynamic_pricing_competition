import pandas as pd
print('Start train script')

import numpy as np
import tensorflow as tf
import datetime
import argparse

from azureml.core.run import Run
from azureml.core import Workspace, Dataset
from tf_agents.environments import tf_py_environment, py_environment, utils
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import EpsilonGreedyPolicy, random_tf_policy
from tf_agents.policies.q_policy import QPolicy
from tf_agents.policies.boltzmann_policy import BoltzmannPolicy
from tf_agents.trajectories import trajectory, Trajectory, PolicyStep, time_step as ts
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.metrics import tf_metrics
from tf_agents.specs import array_spec
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from helper_functions import *
from environment_functions import *


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    iterator = iter(dataset)
    
    for iteration in range(n_iterations):
        current_metrics = []
        
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        
        train_loss = agent.train(trajectories)
        all_train_loss.append(train_loss.loss.numpy())

        for i in range(len(train_metrics)):
            current_metrics.append(train_metrics[i].result().numpy())
            
        all_metrics.append(current_metrics)
        
        if iteration % 1000 == 0:
            print("\nIteration: {}, loss:{:.2f}".format(iteration, train_loss.loss.numpy()))
            
            for i in range(len(train_metrics)):
                print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))
                
                if type(i) == tf_metrics.AverageReturnMetric:
                    run.log('Training avg reward', train_metrics[i].result().numpy())


run = Run.get_context()

# Receive and parse the hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, help='Which policy the agent should use in decision making')
parser.add_argument('--boltzmann-temperature-start', dest='boltzmann_temperature_start', type=float,
                    help='How much exploration to start with (1 high, 0 low)')
parser.add_argument('--boltzmann-temperature-end', dest='boltzmann_temperature_end', type=float,
                    help='How much exploration to end with (1 high, 0 low)')
parser.add_argument('--epsilon-greedy-start', dest='epsilon_greedy_start', type=float,
                    help='How much exploration to start with (1 high, 0 low)')
parser.add_argument('--epsilon-greedy-end', dest='epsilon_greedy_end', type=float, 
                    help='How much exploration to end with (1 high, 0 low)')
parser.add_argument('--update-period', dest='update_period', type=int, help='')
parser.add_argument('--decay-steps', dest='decay_steps', type=int, help='')
parser.add_argument('--min-action', dest='min_action', type=int, help='')
parser.add_argument('--max-action', dest='max_action', type=int, help='')
parser.add_argument('--action-step', dest='action_step', type=int, help='')
parser.add_argument('--comp-sellout-price', dest='comp_sellout_price', type=int, help='')
parser.add_argument('--replay-buffer-max-size', dest='replay_buffer_max_size', type=int, help='')
parser.add_argument('--replay-buffer-batch-size', dest='replay_buffer_batch_size', type=int, help='')
parser.add_argument('--discount', dest='discount', type=float, help='')
parser.add_argument('--sample-batch-size', dest='sample_batch_size', type=int, help='')
parser.add_argument('--num-steps', dest='num_steps', type=int, help='')
parser.add_argument('--train-seasons', dest='train_seasons', type=int, help='')
parser.add_argument('--early-stop-improvement-seasons', dest='early_stop_improvement_seasons', type=int, help='')
parser.add_argument('--early-stopping-patience', dest='early_stopping_patience', type=int, help='')
parser.add_argument('--evaluation-nr-seasons', dest='evaluation_nr_seasons', type=int, help='')
parser.add_argument('--hidden-units-layer1', dest='hidden_units_layer1', type=int, help='')
parser.add_argument('--hidden-units-layer2', dest='hidden_units_layer2', type=int, help='')
parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='')
parser.add_argument('--beta-1', dest='beta_1', type=float, help='')
parser.add_argument('--beta-2', dest='beta_2', type=float, help='')
parser.add_argument('--plot-interval', dest='plot_interval', type=int, help='')
parser.add_argument('--sample-seasons-for-plots', dest='sample_seasons_for_plots', type=int, help='')

args = parser.parse_args()

print('Start logging arguments')
# Log the current params used for this run
for arg in args._get_kwargs():
    run.log(arg[0], arg[1])
    
## Settings

# Env settings
num_actions = (args.max_action-args.min_action) / args.action_step
num_features = 33  # TODO: Make dynamic

# Set seed for reproducability
seed = 123
tf.random.set_seed(seed)

print('Set up the environments')
dpc_game = DynamicPricingCompetition()
simulator = CreateAirlineSimulation()
environment = AirlineEnvironment(dpc_game, simulator, num_features, num_actions, 
                                 args.discount, args.min_action, args.action_step, args.comp_sellout_price)
utils.validate_py_environment(environment, episodes=5)

# Create train and evaluate env
train_env = tf_py_environment.TFPyEnvironment(environment)
eval_env = tf_py_environment.TFPyEnvironment(environment)

print('Set up the agent and the network for the agent')
init = tf.keras.initializers.HeUniform()
layer1 = Dense(units=args.hidden_units_layer1, input_shape=(num_features,), activation='relu', 
               kernel_initializer=init, name='hidden_layer1')
layer2 = Dense(units=args.hidden_units_layer2, activation='relu', kernel_initializer=init, name='hidden_layer2')
layer3 = Dense(units=num_actions, activation=None, kernel_initializer=init)
q_net = sequential.Sequential([layer1, layer2, layer3])

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2)

train_step_counter = tf.Variable(0)

if args.policy == 'boltzmann_temperature':
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=args.boltzmann_temperature_start, 
                    decay_steps=args.decay_steps,
                    end_learning_rate=args.boltzmann_temperature_end)

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=args.discount,
        boltzmann_temperature=lambda: epsilon_fn(train_step_counter),
        epsilon_greedy=None,
        train_step_counter=train_step_counter
    )
elif args.policy == 'epsilon_greedy':
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=args.epsilon_greedy_start, 
                    decay_steps=args.decay_steps,
                    end_learning_rate=args.epsilon_greedy_end)

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=args.discount,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter),
        train_step_counter=train_step_counter
    )

agent.initialize()

# replay buffer and driver for training
replay_buffer = TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=args.replay_buffer_batch_size,
    max_length=args.replay_buffer_max_size
)

replay_buffer_observer = replay_buffer.add_batch
train_metrics = [tf_metrics.AverageReturnMetric()]

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=args.update_period)

print('Initial data generation and setting up the dataset for training')
initial_collect_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

init_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(args.replay_buffer_max_size)],
    num_steps=args.replay_buffer_max_size)

final_time_step, final_policy_state = init_driver.run()

dataset = replay_buffer.as_dataset(sample_batch_size=args.sample_batch_size, num_steps=args.num_steps, num_parallel_calls=4).prefetch(4)

collect_driver.run = common.function(collect_driver.run)
agent.train = common.function(agent.train)

print('Start simulations and training')
all_train_loss = []
all_metrics = []

for seasons_batch in range(args.train_seasons // args.plot_interval):
    train_agent(args.plot_interval * 100)
    
    for i in range(args.sample_seasons_for_plots):
        
        selling_season = (seasons_batch+1)*args.plot_interval -i -1
        competition_id = f'dqnagent{str(selling_season)}'

        plot_price_and_loadfactor(dpc_game.competition_results_df, competition_id, selling_season, 
                                  run, seasons_batch, args.plot_interval, i)
        

# Plot and save the competition results
revenue_ps = dpc_game.competition_results_df.groupby('selling_season')['revenue'].sum().reset_index()

plt.figure(figsize=(16,6))
plt.title('Total revenue per season')
sns.lineplot(x=revenue_ps['selling_season'], y=revenue_ps['revenue'].astype(int))
sns.regplot(x=revenue_ps['selling_season'], y=revenue_ps['revenue'].astype(int), ci=False, scatter=False)
plt.grid()

run.log_image('Performance_plot_per_season', plot=plt)

# Check performance on evaluation environment (policy is now different)
eval_avg_return = compute_avg_return(eval_env, agent.collect_policy, args.evaluation_nr_seasons)
run.log('Evaluation_reward', eval_avg_return)
run.parent.log('Evaluation_reward', eval_avg_return)

# TODO: Save agent
