from Demands.demands import DemandList, Demand
from Node.node import Node
from Parameters import num_vehicles, vehicle_capacity, distance_matrix
import pandas as pd
import math
import random
import copy
from Vehicle.vehicle import Vehicle

class VRPPD:
    def __init__(self, demands: DemandList):
        self.demands = demands
        self.vehicle_list = [Vehicle(i) for i in range(num_vehicles)]
        self.generate_initial_solution()

    def generate_initial_solution(self) -> list[Vehicle]:
        picked_demands : list[Demand] = []
        remaining_demands : list[Demand] = self.demands.get_all_demands()
        while remaining_demands or picked_demands:
            min_distance = float('inf')
            min_vehicle = None
            min_node = None
            min_demand = None
            action = None
            for vehicle in self.vehicle_list:
                for demand in remaining_demands:
                    for pickup_node in demand.get_pickup_nodes():
                        if vehicle.get_route().load[-1] + demand.get_demand_load() <= vehicle_capacity:
                            distance = distance_matrix[vehicle.get_last_node().get_id()][pickup_node.get_id()]
                            if distance < min_distance:
                                min_distance = distance
                                min_vehicle = vehicle
                                min_node = pickup_node
                                min_demand = demand
                                action = "pickup"
                for demand in picked_demands:
                    for delivery_node in demand.get_delivery_nodes():
                        distance = distance_matrix[vehicle.get_last_node().get_id()][delivery_node.get_id()]
                        if distance < min_distance:
                            min_distance = distance
                            min_vehicle = vehicle
                            min_node = delivery_node
                            min_demand = demand
                            action = "delivery"
                min_vehicle.append_node(min_node)
                min_vehicle.update_load()
                min_vehicle.update_time(self.demands)
                if action == "pickup":
                    remaining_demands.remove(min_demand)
                    picked_demands.append(min_demand)
                    min_vehicle.append_demand(min_demand)
                elif action == "delivery":
                    picked_demands.remove(min_demand)
        total_cost = 0
        for vehicle in self.vehicle_list:
            print("vehicle ",vehicle.get_index())
            for node in vehicle.get_route().nodes:
                print(node.get_id()," ",node.get_type()," ",node.get_demand_load())
            vehicle_cost = vehicle.calculate_total_cost(self.demands)
            print("costs: ",vehicle_cost)
            total_cost += vehicle_cost
        print("total cost: ",total_cost)
        return self.vehicle_list
    
def destroy_demands(vehicle_list: list[Vehicle], destroyed_demands: DemandList, demandlist: DemandList):
    for vehicle in vehicle_list:
        for node in vehicle.get_route():
            if demandlist.get_demand_by_node(node) in destroyed_demands:
                vehicle.remove_node(node)
        for demand in vehicle.get_demand_list():
            if demand in destroyed_demands:
                vehicle.remove_demand(demand)
        vehicle.update_load()

def random_destroy(vehicle_list: list[Vehicle], demandlist: DemandList, destruction_rate: float):
    destroy_num = math.ceil(len(DemandList) * destruction_rate)
    destroyed_demands = random.sample(demandlist.get_all_demands(), destroy_num)
    destroy_demands(vehicle_list, destroyed_demands, demandlist)
    
def find_worst_demand(vehicle_list: list[Vehicle], demandlist: DemandList):
    worst_demand = None
    worst_cost = float('inf')
    for vehicle in vehicle_list:
        for demand in vehicle.get_demand_list():
            cost = vehicle.calculate_demand_cost(demand, demandlist)
            if cost < worst_cost:
                worst_cost = cost
                worst_demand = demand
    return worst_demand
            
        
def worst_destroy(vehicle_list: list[Vehicle], demandlist: DemandList, destruction_rate: float):
    destroy_num = math.ceil(len(DemandList) * destruction_rate)
    destroyed_demands = []
    for i in range(destroy_num):
        worst_demand = find_worst_demand(vehicle_list, demandlist)
        destroyed_demands.append(worst_demand)
        destroy_demands(vehicle_list, [worst_demand], demandlist)
    
    
def best_repair(vehicle_list: list[Vehicle], demandlist: DemandList, repairing_demands: list[Demand]):
    new_vehicle_list = copy.deepcopy(vehicle_list)
    possible = True
    while possible:
        best_choice = None
        best_change = None
        best_cost = float('inf')
        for demand in repairing_demands:
            for vehicle in new_vehicle_list:
                best_update, cost = vehicle.insert_demand(demand)
                if best_update:
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = vehicle
                        best_change = best_update
        if best_change:
            for vehicle in new_vehicle_list:
                if vehicle == best_choice:
                    vehicle = best_change
        else:
            possible = False
    return new_vehicle_list

def random_repair(vehicle_list: list[Vehicle], repairing_demands: list[Demand]):
    new_vehicle_list = copy.deepcopy(vehicle_list)
    failure_count = 0
    max_failure = math.ceil(len(repairing_demands) * 0.1)
    while failure_count < max_failure:
        demand = random.choice(repairing_demands)
        vehicle = random.choice(new_vehicle_list)
        new_vehicle = vehicle.random_insert(demand)
        if new_vehicle:
            vehicle = new_vehicle
        else:
            failure_count += 1
    return new_vehicle_list



        
