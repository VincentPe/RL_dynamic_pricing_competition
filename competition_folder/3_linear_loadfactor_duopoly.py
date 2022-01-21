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

    if selling_period_in_current_season == 1:

        capacity = 80
        price = 50

        # Information dump is passed on to the next selling period in the season (not logged)
        information_dump = {
            "Capacity": capacity,
            "Prices": [price],
            "Comp_capacity": [True]
        }

        return (price, information_dump)

    #elif selling_period_in_current_season <= 10:
    else:

        # Update Capacity
        capacity_last_period = information_dump["Capacity"]
        capacity = capacity_last_period - demand_historical_in_current_season[-1]

        #print(f'capacity: {capacity}')

        # Update comp capacity
        comp_capacity = information_dump['Comp_capacity']
        comp_capacity.append(competitor_has_capacity_current_period_in_current_season)

        # Planned loadfactor (for filling uniformly)
        planned_capacity = 80 - selling_period_in_current_season * 0.8

        #print(f'planned_capacity: {planned_capacity}')

        # print(prices_historical_in_current_season[0][-3:])  # Our prices
        # print(prices_historical_in_current_season[1][-3:])  # Comp prices

        # Determine new price based on feeling the demand
        old_price = information_dump['Prices'][-1]
        avg_demand = demand_historical_in_current_season[-3:].sum() / len(demand_historical_in_current_season[-3:])

        #print(f'avg demand {avg_demand}')

        # If we are going to slow, decrease price (but not if we are already selling much recently)
        if (planned_capacity < capacity) & (avg_demand < 2.0):
            price = old_price - 2
        # If we are going to fast, increase price (but not if we are not selling recently anyway)
        elif (planned_capacity > capacity) & (avg_demand > 0.5):
            price = old_price + 2
        else:
            price = old_price

        # If competitor just emptied out stock, raise price
        if information_dump['Comp_capacity'][-1] and not competitor_has_capacity_current_period_in_current_season:
            price += 5

        #print(f'price {price}')

        # Update price
        prices = information_dump['Prices']
        prices.append(price)

        # Information dump is passed on to the next selling period in the season (not logged)
        information_dump = {
            "Capacity": capacity,
            "Prices": prices,
            "Comp_capacity": comp_capacity
        }

        return (price, information_dump)
