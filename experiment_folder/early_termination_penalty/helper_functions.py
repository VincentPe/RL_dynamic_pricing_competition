import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import lines


def plot_price_and_loadfactor(data, competition_id, selling_season, run, seasons_batch, plot_interval, i):
    
    # Create temp dataset for competition id and season
    comp_data = data[
        (data['selling_season'] == selling_season) & 
        (data['competition_id'] == competition_id)
    ].reset_index(drop=True)
    comp_data['loadfactor'] = comp_data['demand'].cumsum() / 80 * 100
    
    # Create plots for price, comp price and loadfactor
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()
    
    lns1 = ax1.plot(comp_data['price'], color='blue', label='Our price')
    lns2 = ax1.plot(comp_data['price_competitor'], color='orange', label='Competitor price')
    lns3 = ax2.plot(comp_data['loadfactor'], color='green', label='Our loadfactor')
    
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Loadfactor (%)')
    ax2.set_ylim(0, 110)
    
    # Add vertical lines where inventory sells out
    if comp_data['loadfactor'].iloc[-1] == 100:
        x = np.where(comp_data['loadfactor'] == 100)[0][0]
        lns4 = lines.Line2D(xdata=[x, x], ydata=[0, 110], color='black', label='Our sellout day')
        lns4 = ax2.add_line(lns4)
        
    if comp_data['competitor_has_capacity'].sum() < 100:
        x = comp_data['competitor_has_capacity'].sum()
        lns5 = lines.Line2D(xdata=[x, x], ydata=[0, 110], color='black', linestyle='--', label='Comp sellout day')
        lns5 = ax2.add_line(lns5)
    
    # Add descriptive info
    plt.title('Price comparison')
    plt.text(.85, .25, 
             (
                 f"Competitor name: {comp_data['competitor_id'][0]} \n " +
                 f"Competition id: {competition_id} \n " +
                 f"Selling season: {selling_season} \n " +
                 f"Total seats sold: {comp_data['demand'].sum()} \n " +
                 f"Total revenue: {(comp_data['demand'] * comp_data['price']).sum()}"
             ),
             bbox={'facecolor':'w','pad':5}, 
             ha="center", va="top", 
             transform=plt.gca().transAxes
            )
    
    # Add labels
    lns = lns1+lns2+lns3
    if comp_data['loadfactor'].iloc[-1] == 100:
        lns = lns+[lns4]
    if comp_data['competitor_has_capacity'].sum() < 100:    
        lns = lns+[lns5]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left")

    ax1.xaxis.grid(True)
    plt.grid()
    
    if run is not None:
        run.log_image(f'trajectory_plot_{(seasons_batch+1) * plot_interval}_seasons_{i}', plot=plt)
    
    
def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
    total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")