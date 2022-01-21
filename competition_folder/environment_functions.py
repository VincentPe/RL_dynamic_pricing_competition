import pandas as pd
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.policies import tf_policy, tf_py_policy, py_policy
from tf_agents.trajectories import Trajectory, PolicyStep, time_step as ts
from tf_agents.specs import array_spec


class DynamicPricingCompetition():
    """
    A game object where the agent can interact with and that we can update remotely
    to adjust the current state based on recent observations.
    This class also keeps track of all the steps and rewards that took place for later analysis.
    """

    def __init__(self):
        self.selling_period = 1
        self.loadfactor = 0
        #         self.comp_loadfactor = 0
        self.competitor_has_capacity = 1
        self.price_competitor_t1 = 50
        self.price_competitor_t2 = 50
        self.price_competitor_t3 = 50
        self.price_competitor_t4 = 50
        self.price_competitor_t5 = 50
        self.price_competitor_t6 = 50
        self.price_competitor_t7 = 50
        self.price_competitor_t8 = 50
        self.price_competitor_t9 = 50
        self.price_competitor_t10 = 50
        self.price_t1 = 50
        self.price_t2 = 50
        self.price_t3 = 50
        self.price_t4 = 50
        self.price_t5 = 50
        self.price_t6 = 50
        self.price_t7 = 50
        self.price_t8 = 50
        self.price_t9 = 50
        self.price_t10 = 50
        self.demand_t1 = 1
        self.demand_t2 = 1
        self.demand_t3 = 1
        self.demand_t4 = 1
        self.demand_t5 = 1
        self.demand_t6 = 1
        self.demand_t7 = 1
        self.demand_t8 = 1
        self.demand_t9 = 1
        self.demand_t10 = 1
        #         self.demand_competitor_t1 = 1
        #         self.demand_competitor_t2 = 1
        #         self.demand_competitor_t3 = 1
        #         self.demand_competitor_t4 = 1
        #         self.demand_competitor_t5 = 1
        #         self.demand_competitor_t6 = 1
        #         self.demand_competitor_t7 = 1
        #         self.demand_competitor_t8 = 1
        #         self.demand_competitor_t9 = 1
        #         self.demand_competitor_t10 = 1
        #         self.competition_results_df = pd.DataFrame(columns=[
        #             'our_strategy',
        #             'competition_id',
        #             'selling_season',
        #             'selling_period',
        #             'competitor_id',
        #             'price_competitor',
        #             'price',
        #             'demand',
        #             'competitor_has_capacity',
        #             'revenue'
        #         ])

        self.state = [
            self.selling_period,
            self.loadfactor,
            #             self.comp_loadfactor,
            self.competitor_has_capacity,
            self.price_competitor_t1,
            self.price_competitor_t2,
            self.price_competitor_t3,
            self.price_competitor_t4,
            self.price_competitor_t5,
            self.price_competitor_t6,
            self.price_competitor_t7,
            self.price_competitor_t8,
            self.price_competitor_t9,
            self.price_competitor_t10,
            self.price_t1,
            self.price_t2,
            self.price_t3,
            self.price_t4,
            self.price_t5,
            self.price_t6,
            self.price_t7,
            self.price_t8,
            self.price_t9,
            self.price_t10,
            self.demand_t1,
            self.demand_t2,
            self.demand_t3,
            self.demand_t4,
            self.demand_t5,
            self.demand_t6,
            self.demand_t7,
            self.demand_t8,
            self.demand_t9,
            self.demand_t10,
            #             self.demand_competitor_t1,
            #             self.demand_competitor_t2,
            #             self.demand_competitor_t3,
            #             self.demand_competitor_t4,
            #             self.demand_competitor_t5,
            #             self.demand_competitor_t6,
            #             self.demand_competitor_t7,
            #             self.demand_competitor_t8,
            #             self.demand_competitor_t9,
            #             self.demand_competitor_t10,
        ]
        self._reward = 0

    def reset(self):
        self.selling_period = 1
        self.loadfactor = 0
        #         self.comp_loadfactor = 0
        self.competitor_has_capacity = 1
        self.price_competitor_t1 = 50
        self.price_competitor_t2 = 50
        self.price_competitor_t3 = 50
        self.price_competitor_t4 = 50
        self.price_competitor_t5 = 50
        self.price_competitor_t6 = 50
        self.price_competitor_t7 = 50
        self.price_competitor_t8 = 50
        self.price_competitor_t9 = 50
        self.price_competitor_t10 = 50
        self.price_t1 = 50
        self.price_t2 = 50
        self.price_t3 = 50
        self.price_t4 = 50
        self.price_t5 = 50
        self.price_t6 = 50
        self.price_t7 = 50
        self.price_t8 = 50
        self.price_t9 = 50
        self.price_t10 = 50
        self.demand_t1 = 1
        self.demand_t2 = 1
        self.demand_t3 = 1
        self.demand_t4 = 1
        self.demand_t5 = 1
        self.demand_t6 = 1
        self.demand_t7 = 1
        self.demand_t8 = 1
        self.demand_t9 = 1
        self.demand_t10 = 1
        #         self.demand_competitor_t1 = 1
        #         self.demand_competitor_t2 = 1
        #         self.demand_competitor_t3 = 1
        #         self.demand_competitor_t4 = 1
        #         self.demand_competitor_t5 = 1
        #         self.demand_competitor_t6 = 1
        #         self.demand_competitor_t7 = 1
        #         self.demand_competitor_t8 = 1
        #         self.demand_competitor_t9 = 1
        #         self.demand_competitor_t10 = 1

        self.state = [
            self.selling_period,
            self.loadfactor,
            #             self.comp_loadfactor,
            self.competitor_has_capacity,
            self.price_competitor_t1,
            self.price_competitor_t2,
            self.price_competitor_t3,
            self.price_competitor_t4,
            self.price_competitor_t5,
            self.price_competitor_t6,
            self.price_competitor_t7,
            self.price_competitor_t8,
            self.price_competitor_t9,
            self.price_competitor_t10,
            self.price_t1,
            self.price_t2,
            self.price_t3,
            self.price_t4,
            self.price_t5,
            self.price_t6,
            self.price_t7,
            self.price_t8,
            self.price_t9,
            self.price_t10,
            self.demand_t1,
            self.demand_t2,
            self.demand_t3,
            self.demand_t4,
            self.demand_t5,
            self.demand_t6,
            self.demand_t7,
            self.demand_t8,
            self.demand_t9,
            self.demand_t10,
            #             self.demand_competitor_t1,
            #             self.demand_competitor_t2,
            #             self.demand_competitor_t3,
            #             self.demand_competitor_t4,
            #             self.demand_competitor_t5,
            #             self.demand_competitor_t6,
            #             self.demand_competitor_t7,
            #             self.demand_competitor_t8,
            #             self.demand_competitor_t9,
            #             self.demand_competitor_t10,
        ]
        self._reward = 0

    def update_state(self, vars_dict):
        self.selling_period = vars_dict['selling_period']
        self.loadfactor = vars_dict['loadfactor']
        #         self.comp_loadfactor = vars_dict['comp_loadfactor']
        self.competitor_has_capacity = vars_dict['competitor_has_capacity']
        self.price_competitor_t1 = vars_dict['price_competitor_t-1']
        self.price_competitor_t2 = vars_dict['price_competitor_t-2']
        self.price_competitor_t3 = vars_dict['price_competitor_t-3']
        self.price_competitor_t4 = vars_dict['price_competitor_t-4']
        self.price_competitor_t5 = vars_dict['price_competitor_t-5']
        self.price_competitor_t6 = vars_dict['price_competitor_t-6']
        self.price_competitor_t7 = vars_dict['price_competitor_t-7']
        self.price_competitor_t8 = vars_dict['price_competitor_t-8']
        self.price_competitor_t9 = vars_dict['price_competitor_t-9']
        self.price_competitor_t10 = vars_dict['price_competitor_t-10']
        self.price_t1 = vars_dict['price_t-1']
        self.price_t2 = vars_dict['price_t-2']
        self.price_t3 = vars_dict['price_t-3']
        self.price_t4 = vars_dict['price_t-4']
        self.price_t5 = vars_dict['price_t-5']
        self.price_t6 = vars_dict['price_t-6']
        self.price_t7 = vars_dict['price_t-7']
        self.price_t8 = vars_dict['price_t-8']
        self.price_t9 = vars_dict['price_t-9']
        self.price_t10 = vars_dict['price_t-10']
        self.demand_t1 = vars_dict['demand_t-1']
        self.demand_t2 = vars_dict['demand_t-2']
        self.demand_t3 = vars_dict['demand_t-3']
        self.demand_t4 = vars_dict['demand_t-4']
        self.demand_t5 = vars_dict['demand_t-5']
        self.demand_t6 = vars_dict['demand_t-6']
        self.demand_t7 = vars_dict['demand_t-7']
        self.demand_t8 = vars_dict['demand_t-8']
        self.demand_t9 = vars_dict['demand_t-9']
        self.demand_t10 = vars_dict['demand_t-10']
        #         self.demand_competitor_t1 = vars_dict['demand_competitor_t-1']
        #         self.demand_competitor_t2 = vars_dict['demand_competitor_t-2']
        #         self.demand_competitor_t3 = vars_dict['demand_competitor_t-3']
        #         self.demand_competitor_t4 = vars_dict['demand_competitor_t-4']
        #         self.demand_competitor_t5 = vars_dict['demand_competitor_t-5']
        #         self.demand_competitor_t6 = vars_dict['demand_competitor_t-6']
        #         self.demand_competitor_t7 = vars_dict['demand_competitor_t-7']
        #         self.demand_competitor_t8 = vars_dict['demand_competitor_t-8']
        #         self.demand_competitor_t9 = vars_dict['demand_competitor_t-9']
        #         self.demand_competitor_t10 = vars_dict['demand_competitor_t-10']

        self.state = [
            self.selling_period,
            self.loadfactor,
            #             self.comp_loadfactor,
            self.competitor_has_capacity,
            self.price_competitor_t1,
            self.price_competitor_t2,
            self.price_competitor_t3,
            self.price_competitor_t4,
            self.price_competitor_t5,
            self.price_competitor_t6,
            self.price_competitor_t7,
            self.price_competitor_t8,
            self.price_competitor_t9,
            self.price_competitor_t10,
            self.price_t1,
            self.price_t2,
            self.price_t3,
            self.price_t4,
            self.price_t5,
            self.price_t6,
            self.price_t7,
            self.price_t8,
            self.price_t9,
            self.price_t10,
            self.demand_t1,
            self.demand_t2,
            self.demand_t3,
            self.demand_t4,
            self.demand_t5,
            self.demand_t6,
            self.demand_t7,
            self.demand_t8,
            self.demand_t9,
            self.demand_t10,
            #             self.demand_competitor_t1,
            #             self.demand_competitor_t2,
            #             self.demand_competitor_t3,
            #             self.demand_competitor_t4,
            #             self.demand_competitor_t5,
            #             self.demand_competitor_t6,
            #             self.demand_competitor_t7,
            #             self.demand_competitor_t8,
            #             self.demand_competitor_t9,
            #             self.demand_competitor_t10,
        ]
        self._reward = 0

    def update_reward(self, reward):
        self.reward = reward


# Environment in which the agent operates in, and is protected from altering
class AirlineEnvironment(py_environment.PyEnvironment):

    def __init__(self, dpc_game, simulator, num_features, num_actions, discount, min_action,
                 action_step, comp_sellout_price, early_termination_penalty=0, price_diff_penalty=0,
                 loadfactor_diff_penalty=0, stock_remainder_penalty=0):
        """
        Initialize what actions the agent can take,
        and what the observation space will look like.

        Also initialize the environment where the agent will interact with.
        """
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_features,), dtype=np.int32, name='observation'
        )
        self._episode_ended = False
        self._discount = discount
        self._dpc_game = dpc_game
        self._simulator = simulator
        self._min_action = min_action
        self._action_step = action_step
        self._comp_sellout_price = comp_sellout_price
        self._early_termination_penalty = early_termination_penalty
        self._price_diff_penalty = price_diff_penalty
        self._loadfactor_diff_penalty = loadfactor_diff_penalty
        self._stock_remainder_penalty = stock_remainder_penalty

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def current_time_step(self):
        return self._current_time_step

    def reset(self):
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        self._current_time_step = self._step(action)
        return self._current_time_step

    def _reset(self):
        self._episode_ended = False
        self._dpc_game.reset()
        self._simulator.reset_environment()
        return ts.restart(np.array(self._dpc_game.state, dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        self._our_demand = self._simulator.get_demand()

        self._reward = self._our_demand * self._simulator._our_price

        # Update states
        vars_dict = {
            'selling_period': self._simulator._selling_period,
            'loadfactor': 80 - self._simulator._our_stock,
            #             'comp_loadfactor': 80 - self._simulator.comp_stock,
            'competitor_has_capacity': self._simulator._competitor_has_capacity,
            'price_competitor_t-1': self._simulator._comp_price,
            'price_competitor_t-2': self._dpc_game.price_competitor_t1,
            'price_competitor_t-3': self._dpc_game.price_competitor_t2,
            'price_competitor_t-4': self._dpc_game.price_competitor_t3,
            'price_competitor_t-5': self._dpc_game.price_competitor_t4,
            'price_competitor_t-6': self._dpc_game.price_competitor_t5,
            'price_competitor_t-7': self._dpc_game.price_competitor_t6,
            'price_competitor_t-8': self._dpc_game.price_competitor_t7,
            'price_competitor_t-9': self._dpc_game.price_competitor_t8,
            'price_competitor_t-10': self._dpc_game.price_competitor_t9,
            'price_t-1': self._simulator._our_price,
            'price_t-2': self._dpc_game.price_t1,
            'price_t-3': self._dpc_game.price_t2,
            'price_t-4': self._dpc_game.price_t3,
            'price_t-5': self._dpc_game.price_t4,
            'price_t-6': self._dpc_game.price_t5,
            'price_t-7': self._dpc_game.price_t6,
            'price_t-8': self._dpc_game.price_t7,
            'price_t-9': self._dpc_game.price_t8,
            'price_t-10': self._dpc_game.price_t9,
            'demand_t-1': self._our_demand,
            'demand_t-2': self._dpc_game.demand_t1,
            'demand_t-3': self._dpc_game.demand_t2,
            'demand_t-4': self._dpc_game.demand_t3,
            'demand_t-5': self._dpc_game.demand_t4,
            'demand_t-6': self._dpc_game.demand_t5,
            'demand_t-7': self._dpc_game.demand_t6,
            'demand_t-8': self._dpc_game.demand_t7,
            'demand_t-9': self._dpc_game.demand_t8,
            'demand_t-10': self._dpc_game.demand_t9,
            #             'demand_competitor_t-1': self._comp_demand[0],
            #             'demand_competitor_t-2': self._dpc_game.demand_competitor_t1,
            #             'demand_competitor_t-3': self._dpc_game.demand_competitor_t2,
            #             'demand_competitor_t-4': self._dpc_game.demand_competitor_t3,
            #             'demand_competitor_t-5': self._dpc_game.demand_competitor_t4,
            #             'demand_competitor_t-6': self._dpc_game.demand_competitor_t5,
            #             'demand_competitor_t-7': self._dpc_game.demand_competitor_t6,
            #             'demand_competitor_t-8': self._dpc_game.demand_competitor_t7,
            #             'demand_competitor_t-9': self._dpc_game.demand_competitor_t8,
            #             'demand_competitor_t-10': self._dpc_game.demand_competitor_t9,
        }

        self._dpc_game.update_state(vars_dict)
        self._dpc_game.update_reward(self._reward)

        # Make sure episodes don't go on forever.
        if self._dpc_game.state[0] == 100:
            # Add additional penalty for ending the season with high stock left
            self._episode_ended = True
            return ts.termination(
                np.array(self._dpc_game.state, dtype=np.int32),
                self._reward - self._stock_remainder_penalty * self._simulator._our_stock
            )
        elif self._dpc_game.loadfactor >= 80:
            # Add additional penalty for ending the season early (higher penalty longer in advance)
            self._episode_ended = True
            return ts.termination(
                np.array(self._dpc_game.state, dtype=np.int32),
                self._reward - self._early_termination_penalty * (100 - self._dpc_game.state[0])
            )
        else:
            # Add additional penalty for changing prices by a lot
            price_diff_penal = abs(self._simulator._our_price - self._dpc_game.price_t1) ** 2 * self._price_diff_penalty
            # Add additional penaly for selling out too quickly
            load_diff = abs(self._dpc_game.loadfactor - (0.8 * self._dpc_game.state[0]))
            load_diff_penalty = load_diff ** 2 * self._loadfactor_diff_penalty

            return ts.transition(
                np.array(self._dpc_game.state, dtype=np.int32),
                reward=self._reward - price_diff_penal - load_diff_penalty,
                discount=self._discount
            )


class ExternalSimulator():

    def __init__(self):
        self._selling_period = 1
        self._our_stock = 80
        self._our_price = 50
        self._our_comp_price = 50
        self._competitor_has_capacity = True
        self._last_demand = 1

    def get_demand(self):
        return self._last_demand

    def update(self, last_price, last_comp_price, last_demand, competitor_has_capacity):
        self._selling_period += 1
        self._our_stock -= last_demand
        self._our_price = last_price
        self._comp_price = last_comp_price
        self._last_demand = last_demand
        self._competitor_has_capacity = competitor_has_capacity

    def reset_environment(self):
        self._selling_period = 1
        self._our_stock = 80
        self._our_price = 50
        self._comp_price = 50
        self._competitor_has_capacity = True
        self._last_demand = 1


