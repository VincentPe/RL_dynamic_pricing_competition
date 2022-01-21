import numpy as np


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
    price_increment = (70 - 40) / 99
    if selling_period_in_current_season == 1:

        capacity = 80
        price = 40

        # Information dump is passed on to the next selling period in the season (not logged)
        information_dump = {
            "Capacity": capacity,
            "Period_Capacity_Empty": False,
        }

        return (price, information_dump)

    else:
        # Update Capacity
        capacity_last_period = information_dump["Capacity"]
        new_capacity = capacity_last_period - demand_historical_in_current_season[-1]

        # Get Stockout Period where capacity became 0
        capacity_empty = information_dump["Period_Capacity_Empty"]

        # Check if our capacity dropped to zero and if so save the period where this happened
        if new_capacity <= 0 and capacity_empty == False:
            stockout_period = selling_period_in_current_season
        else:
            stockout_period = capacity_empty

        # Update information dump
        information_dump["Capacity"] = new_capacity
        information_dump["Period_Capacity_Empty"] = stockout_period

        # Update the price
        price = 40 + price_increment * (selling_period_in_current_season-1)
        #price = 50

        return (price, information_dump)
