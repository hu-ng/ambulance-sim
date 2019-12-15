import networkx as nx
import random as rd
import matplotlib.pyplot as plt
from collections import deque
import heapq as hq
import numpy as np

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
        """Go through every location, roll random for requests"""
        for node in self.map.nodes:
            if rd.random() < self.map.nodes[node]["prob_call"]:
                request = Request(at_node=node)
                self.send_to_nearest_station(request)


    def find_nearest_station(self, from_node):
        """ Find nearest node with a station from a node using BFS"""
        to_visit = deque([from_node])
        while to_visit:
            curr_node = to_visit.pop()
            if self.map.nodes[curr_node]["station"]:
                return curr_node
            for node in list(self.map.neighbors(curr_node)):
                to_visit.appendleft(node)


    def send_to_nearest_station(self, request):
        """Send request to the nearest station"""
        station_node = self.find_nearest_station(request.location)
        self.map.nodes[station_node]["station"].take_request(request)


    def add_station_to(self, node, num_ambulances):
        if not self.map.nodes[node]["station"]:
            self.map.nodes[node]["station"] = Station(map=self.map, at_node=node, num_ambulances=num_ambulances)
            self.nodes_with_station.append(node)


    def init_stations_amb_1(self, num_stations, total_ambulances):
        """
        Initialize the stations. Prioritize high-risk neighborhoods.
        Equally allocate ambulances to each station
        """
        sorted_risk = [(node, self.map.nodes[node]["prob_call"]) for node in list(self.map.nodes)]
        sorted_risk = sorted(sorted_risk, key=lambda x: x[1], reverse=True)
        for i in range(0, num_stations):
            self.add_station_to(sorted_risk[i][0], total_ambulances // num_stations)

        # Add any leftover ambulances to the first station (most risky location)
        first_station = self.map.nodes[sorted_risk[0][0]]["station"]
        first_station.ambulances += [0] * (total_ambulances % num_stations)


    def use_stations(self):
        """Order all stations to dispatch ambulances"""
        for node in self.nodes_with_station:
            self.map.nodes[node]["station"].dispatch_ambulance()


    def decrement_all_ambulance_wait_time(self):
        """Decrement wait times for all stations"""
        for node in self.nodes_with_station:
            self.map.nodes[node]["station"].decrement_ambulance_wait_time()

    def increment_all_requests_wait_time(self):
        for node in self.nodes_with_station:
            self.map.nodes[node]["station"].increment_requests_wait_time() 


class Station():
    def __init__(self, map, at_node, num_ambulances):
        # Receive calls in a double-ended queue for O(1) updates
        self.requests = deque()
        # Use a heap to always choose the ambulance with the smallest waiting time
        # All ambulances in the beginning are idle
        self.ambulances = [0 for _ in range(num_ambulances)]
        hq.heapify(self.ambulances)
        self.location = at_node
        self.request_wait_times = []
        self.map = map


    def take_request(self, request):
        self.requests.appendleft(request)


    def dispatch_ambulance(self):
        def _sum_of_weights(path):
            """ Helper function to get path total weight"""
            sum = 0
            for i in range(len(path) - 1):
                sum += self.map.edges[(path[i], path[i+1])]["weight"]
            return sum

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
                    patient_wait_time = _sum_of_weights(path) + rd.randint(5, 10)
                    ambulance_wait_time = patient_wait_time + _sum_of_weights(path)

                # Update ambulance waiting time
                hq.heappushpop(self.ambulances, ambulance_wait_time)

                # Update time patients have to wait
                # The maximum between the time it takes for the ambulance to arrive and
                # the time the caller had to wait for the ambulance arrive
                self.request_wait_times.append(max(patient_wait_time, request.wait_time))


    def decrement_ambulance_wait_time(self):
        """
        Decrease waiting times of all operating vehicles by 1
        """
        def decrement(wait_time):
            return wait_time - 1 if wait_time > 0 else wait_time
        
        # print("node", self.location, self.ambulances)
        self.ambulances = list(map(decrement, self.ambulances))
        hq.heapify(self.ambulances)
    

    def increment_requests_wait_time(self):
        """Increment wait time for all unserved calls"""
        for req in self.requests:
            req.increment_wait_time()


class Request():
    def __init__(self, at_node):
        self.location = at_node
        self.wait_time = 0

    def increment_wait_time(self):
        self.wait_time += 1


class Simulation():
    def __init__(self, stations, ambulances):
        self.map = Map()
        self.map.init_stations_amb_1(num_stations=stations, total_ambulances=ambulances)


    def run_sim(self, interval=10):
        for minute in range(20000):
            self.map.decrement_all_ambulance_wait_time()
            self.map.increment_all_requests_wait_time()
            if minute % 10 == 0:
                print(minute)
                self.map.generate_request()
            self.map.use_stations()
    
    def hist_waiting_time(self):
        data = []
        for node in self.map.nodes_with_station:
            data.extend(self.map.map.nodes[node]["station"].request_wait_times)
        plt.hist(data)
        plt.title(f"Average {round(np.average(data), 2)}, Median {round(np.median(data), 2)}, 95% Interval {np.percentile(data, [2.5, 97.5])}")
        plt.show()
        # print(np.average(data))
        # print(np.percentile(data, [2.5, 97.5]))
        # print(np.median(data))

    
sim = Simulation(stations=4, ambulances=20)
sim.run_sim()
sim.hist_waiting_time()
