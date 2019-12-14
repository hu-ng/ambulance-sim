import networkx as nx
import random as rd
import matplotlib.pyplot as plt
from collections import deque
import heapq as hq

map_hanoi = {
    "hoan_kiem": 1,
    "ba_dinh": 2,
    "cau_giay": 3,
    "hai_ba_trung": 4,
    "dong_da": 5,
    "thanh_xuan": 6,
    "gia_lam": 7
}

prob_call_map = {
    "hoan_kiem": 0.8,
    "ba_dinh": 0.5,
    "cau_giay": 0.6,
    "hai_ba_trung": 0.5,
    "dong_da": 0.7,
    "thanh_xuan": 0.4,
    "gia_lam": 0.4
}

edge_hanoi = [
    (map_hanoi['hoan_kiem'], map_hanoi['ba_dinh'], 15),
    (map_hanoi['hoan_kiem'], map_hanoi['dong_da'], 15),
    (map_hanoi['hoan_kiem'], map_hanoi['hai_ba_trung'], 20),
    (map_hanoi['dong_da'], map_hanoi['thanh_xuan'], 15),
    (map_hanoi['dong_da'], map_hanoi['cau_giay'], 25),
    (map_hanoi['dong_da'], map_hanoi['ba_dinh'], 10),
    (map_hanoi['dong_da'], map_hanoi['hai_ba_trung'], 15),
    (map_hanoi['hai_ba_trung'], map_hanoi['thanh_xuan'], 15),
    (map_hanoi['cau_giay'], map_hanoi['thanh_xuan'], 25),
    (map_hanoi['cau_giay'], map_hanoi['ba_dinh'], 15),
    (map_hanoi['ba_dinh'], map_hanoi['gia_lam'], 20),
    (map_hanoi['hoan_kiem'], map_hanoi['gia_lam'], 15),
]


class Map():
    def __init__(self, map=map_hanoi, edge=edge_hanoi, prop_call=prob_call_map):
        self.map = nx.Graph()
        self.map.add_nodes_from([(map_hanoi[k], {"prob_call": prop_call[k], "station": None}) for k in map_hanoi])
        self.map.add_weighted_edges_from(edge_hanoi)
        self.nodes_with_station = []


    def draw_map(self):
        nx.draw(self.map)
        plt.show()


    def generate_request(self):
        """
        Go through every location, roll random for requests
        """
        for node in self.map.nodes:
            if rd.random() < self.map.nodes[node]["prob_call"]:
                request = Request(at_node=node)
                send_to_nearest_station(self, request)


    def find_nearest_station(self, from_node):
        """
        Find nearest node with a station from a node using BFS
        """
        to_visit = deque([from_node])
        while to_visit:
            curr_node = to_visit.pop()
            if self.map[curr_node]["station"]:
                return curr_node
            for node in list(self.map.neighbors(curr_node)):
                to_visit.appendleft(node)


    def send_to_nearest_station(self, request):
        """
        Send request to the nearest station
        """
        station_node = self.find_nearest_station(request.at_node)
        station_node["station"].take_request(request)


    def add_station_to(self, node, num_ambulances):
        if not self.map.node[node]["station"]:
            self.map.node[node]["station"] = Station(map=self.map, at_node=node, num_ambulances=num_ambulances)
            self.nodes_with_station.append(node)


class Station():
    def __init__(self, map, at_node, num_ambulances):
        # Receive calls in a double-ended queue for O(1) updates
        self.requests = deque()
        # Use a heap to always choose the ambulance with the smallest waiting time
        # All ambulances in the beginning are idle
        self.ambulances = hq.heapify([0 for _ in range(num_ambulances)])
        self.location = at_node
        self.request_wait_times = []
        self.map = map


    def take_request(self, request):
        self.requests.appendleft(request)


    def dispatch_ambulance(self):
        # If there is at least 1 request
        if len(self.requests) >= 1:
            # If there are still idle ambulances
            while self.ambulances[0] == 0:
                # In the process of dispatching, if there are no more requests, break
                if len(self.requests) == 0:
                    break
                request = self.requests.pop()

                # If request location is the same as station location
                if request.location == self.location:
                    # Assume it takes no time to get to patient
                    # Only wait to treat patient
                    patient_wait_time = ambulance_wait_time = rd.randint(5, 10)
                else:
                    # Find shortest path given the weights
                    path = nx.shortest_path(self.map, self.location, request.location, weight="weight")
                    patient_wait_time = sum_of_weights(path) + rd.randint(5, 10)
                    ambulance_wait_time = patient_wait_time + sum_of_weights(path)

                # Update ambulance waiting time
                hq.heappushpop(self.ambulances, ambulance_wait_time)

                # Update time patients have to wait
                # The maximum between the time it takes for the ambulance to arrive and
                # the time the caller had to wait for the ambulance arrive
                self.request_wait_times.append(max(patient_wait_time, request.wait_time))



    def sum_of_weights(self, path):
        sum = 0
        for i in range(len(path) - 1):
            sum += self.map.edges[(path[i], path[i+1])]
        return sum


    def decrement_ambulance_wait_time(self):
        """
        Decrease waiting times of all operating vehicles by 1
        """
        self.ambulances = hq.heapify([amb -= 1 for amb in self.ambulances if amb > 0])


class Request():
    def __init__(self, at_node):
        self.location = at_node
        self.wait_time = 0

    def increment_wait_time(self):
        self.wait_time += 1


map = Map()
for edge in map.map.edges:
    print(map.map.edges[(1,2)]["weight"])
