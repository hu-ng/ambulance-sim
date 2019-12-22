import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("mc_data.csv")

def graph(strategy, metric):
    """
    Graph a 3d plot with X: stations, Y: ambulances, Z: metric
    Metric: "completion_time", "outstanding_req", "idle_ambulances"
    """
    # Grab data for the selected strategy
    strat_selector = data["strategy"] == strategy
    strat_data = data[strat_selector]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(strat_data["stations"], strat_data["total_ambulances"], strat_data[metric], cmap='viridis')
    ax.set_xlabel("Number of stations")
    ax.set_ylabel("Number of ambulances")
    ax.set_zlabel(f"{metric}")
    plt.suptitle(f"Surface plot for strategy {strategy} using {metric} metric")
    plt.show()

# graph(1, "completion_time")

# Use this chunk of code to sort through the data and find out combinations of code according to a certain criteria
strat = data["strategy"] == 1
min_completion = data["completion_time"] < 11
print(data[strat].sort_values(by="idle_ambulances", ascending=True))