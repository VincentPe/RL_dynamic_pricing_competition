import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import lines


def plot_price_and_loadfactor(data, competition_id, selling_season, run, eval_env=False):
    
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
        if eval_env:
            run.log_image(f'trajectory_plots/eval_trajectory_plot_selling_season_{int(selling_season)}', plot=plt)
        else:
            run.log_image(f'trajectory_plots/train_trajectory_plot_selling_season_{int(selling_season)}', plot=plt)
            
            
def plot_qlogits_heatmap(qpolicy, eval_env, dpc_game, seasons_batch, i, run=None):
    logits_df = pd.DataFrame()
    time_step = eval_env.reset()

    i=-1
    while int(time_step.step_type) < 2:
        i+=1
        distribution_step = qpolicy.distribution(time_step)
        action_step = qpolicy.action(time_step)

        state_logits = pd.DataFrame({
            'timestep': np.repeat(i+1, 40),
            'action': [x for x in range(40)],
            'logit': distribution_step.action.logits[0]
        })

        logits_df = logits_df.append(state_logits)
        time_step = eval_env.step(action_step)
        
    qactions = logits_df.groupby('timestep').agg({'logit': 'idxmax'}).reset_index()
    
    latest_comp_results = dpc_game.competition_results_df['selling_season'].unique()[-1]
    selling_season = latest_comp_results
    competition_id = f'dqnagent{str(int(selling_season))}'

    # Create temp dataset for competition id and season
    comp_data = dpc_game.competition_results_df[
        (dpc_game.competition_results_df['selling_season'] == selling_season) & 
        (dpc_game.competition_results_df['competition_id'] == competition_id)
    ].reset_index(drop=True)
    comp_data['loadfactor'] = comp_data['demand'].cumsum() / 80 * 100

    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(logits_df.pivot('action', 'timestep', 'logit'), cmap=sns.color_palette("Spectral", as_cmap=True))
    sns.lineplot(x=qactions['timestep'], y=qactions['logit'] + 0.5, color='black', linestyle='--')
    ax.invert_yaxis()
    
    sns.lineplot(x=comp_data['selling_period'], y=(comp_data['price_competitor'].astype(int) -20) / 3 + 0.5, 
             color='purple', linestyle='--', linewidth=2)

    ax2 = plt.twinx()
    sns.lineplot(x=comp_data['selling_period'], y=comp_data['loadfactor'], ax=ax2, color='black', linewidth=2)
    sns.lineplot(x=[x for x in range(len(comp_data))], y=[x for x in range(len(comp_data))], color='gray', linewidth=2)

    ax2.set_ylim(0, 101)

    # Add vertical lines where inventory sells out
    if comp_data['loadfactor'].iloc[-1] == 100:
        plt.axvline(np.where(comp_data['loadfactor'] == 100)[0][0] + 0.5, 0, 39, color='black', linewidth=2)

    if comp_data['competitor_has_capacity'].sum() < 100:
        plt.axvline(comp_data['competitor_has_capacity'].sum() + 0.5, 0, 39, color='purple', linewidth=2)
    
    if run is not None:
        run.log_image(f'qlogits_heatmaps/qlogits_heatmap_{seasons_batch}_{i}', plot=plt)
    
    
def compute_avg_return(environment, dpc_game, policy, num_episodes=10):
    
    for _ in range(num_episodes):
        time_step = environment.reset()

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)

    return dpc_game.competition_results_df[
        dpc_game.competition_results_df['selling_season'] > dpc_game.competition_results_df['selling_season'].max() - num_episodes
    ].groupby('selling_season')['revenue'].sum().mean()


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
            

class ResetPolicy:
    def __init__(self, train_policy):
        self.train_policy = train_policy
        
    def __call__(self, trajectory):
        if trajectory.is_boundary():
            self.train_policy.reset()
