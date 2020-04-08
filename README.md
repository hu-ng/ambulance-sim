# Emergency Ambulance Response Simulation in Python

For this project, I wanted to simulate how different placements of ambulances would affect the emergency response times in a metro area (I chose Hanoi). To do so, I built an object-oriented simulation using Python with the following classes:
- `Map`: Holds the graph structure of city, created using the [networkx](https://networkx.github.io/) library, where each node is a district. Each node can have a `Station` object. Responsible for initializing station placements and distributing ambulances to each station.
- `Station`: Keeps track of the status of emergency calls, dispatches ambulances, and monitors ambulances' travel time (in time step).
- `Request`: Represents a single emergency call. Keeps track of total amount of time step elapsed since creation.
- `Simulation`: Dedicated class to control the flow of the simulation and to extract relevant data for visualization.
