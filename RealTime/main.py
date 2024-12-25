from Demands.demands import DemandList
from Routes.route import Route
from Node.node import Node
from Parameters import num_vehicles, vehicle_capacity, distance_matrix
import numpy as np
import pandas as pd
from Vehicle.vehicle import Vehicle

class VRPPD:
    def __init__(self, demands: DemandList):
        self.demands = demands
        self.vehicle_list = [Vehicle() for _ in range(num_vehicles)]

    def generate_initial_solution(self) -> list[Route]:
        remaining_pickups = self.demands.get_pickup_nodes_list()
        remaining_deliveries = self.demands.get_delivery_nodes_list()
        while remaining_pickups or remaining_deliveries:
            min_distance = float('inf')
            for vehicle in self.vehicle_list:
                for pickup_demand in remaining_pickups:
                    for pickup_node in pickup_demand:
                        if vehicle.get_load() + self.demands.get_demand_by_node(pickup_node).get_demand_load() <= vehicle_capacity:
                            distance = distance_matrix[vehicle.get_current_location().get_id()][pickup_node.get_id()]
                            if distance < min_distance:
                                min_distance = distance
                                min_vehicle = vehicle
                                min_node = pickup_node
                                min_demand = pickup_demand
                                action = "pickup"
                for delivery_demand in remaining_deliveries:
                    for delivery_node in delivery_demand:
                        distance = distance_matrix[vehicle.get_current_location().get_id()][delivery_node.get_id()]
                        if distance < min_distance:
                            min_distance = distance
                            min_vehicle = vehicle
                            min_node = delivery_node
                            min_demand = delivery_demand
                            action = "delivery"
                min_vehicle.append_node(min_node)
                min_vehicle.update_load(min_node)
                if action == "pickup":
                    remaining_pickups.remove(min_demand)
                elif action == "delivery":
                    remaining_deliveries.remove(min_demand)

        return self.vehicle_list
