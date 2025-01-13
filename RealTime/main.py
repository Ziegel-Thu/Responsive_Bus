from Demands.demands import DemandList, Demand
from Node.node import Node
from Parameters import num_vehicles, vehicle_capacity, distance_matrix, initial_temperature, cooling_rate, max_iterations
import pandas as pd
import math
import random
import copy
from Vehicle.vehicle import Vehicle

class VRPPD:
    def __init__(self, demands: DemandList, start_node_id: int, end_node_id: int):
        self.demands = demands
        self.vehicle_list = [Vehicle(i, start_node_id, end_node_id) for i in range(num_vehicles)]
        self.vehicle_list = self.generate_initial_solution()

    def generate_initial_solution(self) -> list[Vehicle]:
        picked_demands : dict[Vehicle, list[Demand]] = {vehicle: [] for vehicle in self.vehicle_list}
        remaining_demands : list[Demand] = copy.deepcopy(self.demands.get_all_demands())
        vehicle_clock = {vehicle: 0 for vehicle in self.vehicle_list}
        Done = False
        while not Done:
            min_time = float('inf')
            min_vehicle = None
            for vehicle in self.vehicle_list:
                print(vehicle_clock[vehicle], vehicle.get_index())
                if remaining_demands or picked_demands[vehicle]:
                    if vehicle_clock[vehicle] < min_time:
                        min_time = vehicle_clock[vehicle]
                        min_vehicle = vehicle
            
            if min_vehicle:
                min_node = None
                min_demand = None
                min_distance = float('inf')
                action = None
                if min_vehicle.is_route_empty():
                    for demand in remaining_demands:
                        for node in demand.get_pickup_nodes():
                            if distance_matrix[min_vehicle.get_start_node_id()][node.get_id()] < min_distance:
                                min_distance = distance_matrix[min_vehicle.get_start_node_id()][node.get_id()]
                                min_node = node
                                min_demand = demand
                                action = "pickup"
                else:
                    for demand in picked_demands[min_vehicle]:
                        for node in demand.get_delivery_nodes():
                            if distance_matrix[min_vehicle.get_route().nodes[-1].get_id()][node.get_id()] < min_distance:
                                min_distance = distance_matrix[min_vehicle.get_route().nodes[-1].get_id()][node.get_id()]
                                min_node = node
                                min_demand = demand
                                action = "delivery"
                    if action == "delivery":
                        for demand in remaining_demands:
                            for node in demand.get_pickup_nodes():
                                if distance_matrix[min_vehicle.get_route().nodes[-1].get_id()][node.get_id()] < min_distance:
                                    if min_vehicle.get_load()[-1] + node.get_demand_load() <= vehicle_capacity:
                                        min_distance = distance_matrix[min_vehicle.get_route().nodes[-1].get_id()][node.get_id()]
                                        min_node = node
                                        min_demand = demand
                                        action = "pickup"
                if action == "delivery":
                    min_vehicle.append_node(min_node)
                    min_vehicle.append_demand(min_demand)
                    picked_demands[min_vehicle].remove(min_demand)
                elif action == "pickup":
                    min_vehicle.append_demand(min_demand)
                    min_vehicle.append_node(min_node)
                    picked_demands[min_vehicle].append(min_demand)
                    remaining_demands.remove(min_demand)
                min_vehicle.update_load()
                vehicle_clock[min_vehicle] += min_distance
                
            else:
                Done = True
        for vehicle in self.vehicle_list:
            vehicle.update_time(self.demands)
        total_cost = 0
        for vehicle in self.vehicle_list:
            print("vehicle ",vehicle.get_index())
            for node in vehicle.get_route().nodes:
                print(node.get_demand_index()," ",node.get_id()," ",node.get_type()," ",node.get_demand_load())
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

def random_destroy(vehicle_list: list[Vehicle], destroyable_demands: DemandList, destruction_rate: float):
    destroy_num = math.ceil(len(destroyable_demands) * destruction_rate)
    destroyed_demands = random.sample(destroyable_demands.get_all_demands(), destroy_num)
    destroy_demands(vehicle_list, destroyed_demands, destroyable_demands)
    
def find_worst_demand(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: DemandList):
    worst_demand = None
    worst_cost = float('inf')
    for vehicle in vehicle_list:
        for demand in destroyable_demands:
            cost = vehicle.calculate_demand_cost(demand, demandlist)
            if cost < worst_cost:
                worst_cost = cost
                worst_demand = demand
    return worst_demand
            
        
def worst_destroy(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: DemandList, destruction_rate: float):
    destroy_num = math.ceil(len(destroyable_demands) * destruction_rate)
    destroyed_demands = []
    for i in range(destroy_num):
        worst_demand = find_worst_demand(vehicle_list, demandlist, destroyable_demands)
        if worst_demand:
            destroyed_demands.append(worst_demand)
            destroy_demands(vehicle_list, [worst_demand], destroyable_demands)
    return destroyed_demands
    
    
def best_repair(vehicle_list: list[Vehicle], demandlist: DemandList, repairing_demands: list[Demand]):
    new_vehicle_list = copy.deepcopy(vehicle_list)
    possible = True
    while possible:
        best_choice = None
        best_change = None
        best_cost = float('inf')
        for demand in repairing_demands:
            for vehicle in new_vehicle_list:
                best_update, cost = vehicle.insert_demand(demand,demandlist)
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
    demands_under_repair = copy.deepcopy(repairing_demands)
    failure_count = 0
    max_failure = math.ceil(len(repairing_demands) * 0.1)
    success = False
    while failure_count < max_failure and demands_under_repair:
        demand = random.choice(demands_under_repair)
        vehicle = random.choice(new_vehicle_list)
        new_vehicle = vehicle.random_insert(demand)
        if new_vehicle:
            vehicle = new_vehicle
            demands_under_repair.remove(demand)
            success = True
        else:
            failure_count += 1
            success = False
    return new_vehicle_list, success

def calculate_cost(vehicle_list: list[Vehicle], demands: DemandList):
    total_cost = 0
    for vehicle in vehicle_list:
        total_cost += vehicle.calculate_total_cost(demands)
    return total_cost

def print_solution(vehicle_list: list[Vehicle], demands: DemandList):
    for vehicle in vehicle_list:
        print("vehicle ",vehicle.get_index())
        for node in vehicle.get_route().nodes:
            print(node.get_demand_index()," ",node.get_type()," ",node.get_demand_load())
        vehicle_cost = vehicle.calculate_total_cost(demands)
        print("costs: ",vehicle_cost)
    total_cost = calculate_cost(vehicle_list, demands)
    print("total cost: ",total_cost)
    
def ALNS(initial_demands: DemandList, new_demands: DemandList, max_iterations: int, start_node: int, end_node: int):
    vrppd = VRPPD(initial_demands, start_node, end_node)
    vehicle_list = vrppd.vehicle_list
    demands = copy.deepcopy(initial_demands)
    for demand in new_demands.get_all_demands():
        demands.add_demand(demand)
    repairable = True
    while repairable:
        new_vehicle_list, success = random_repair(vehicle_list, new_demands)
        if success:
            vehicle_list = new_vehicle_list
        else:
            repairable = False
    iteration = 0
    need_random_destroy = False
    current_solution = vehicle_list
    temporary_best_solution = current_solution
    temporary_best_cost = calculate_cost(temporary_best_solution, demands)
    destroy_rate = 0.1
    temperature = initial_temperature
    best_cost = temporary_best_cost
    best_solution = temporary_best_solution
    
    while iteration < max_iterations:
        if need_random_destroy:
            random_destroy(current_solution, demands, destroy_rate)
            need_random_destroy = False
        else:
            worst_destroy(current_solution, demands, demands, destroy_rate)
            need_random_destroy = True
            
        new_vehicle_list = best_repair(current_solution, demands, demands.get_all_demands())
        if new_vehicle_list:
            current_solution = new_vehicle_list
        
        iteration += 1
        current_cost = calculate_cost(current_solution, demands)
        cost_diff = current_cost - temporary_best_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            temporary_best_solution = current_solution
            temporary_best_cost = current_cost
            print("new solution accepted")
            print_solution(temporary_best_solution, demands)
        if temporary_best_cost < best_cost:
            best_cost = temporary_best_cost
            best_solution = temporary_best_solution
        temperature *= cooling_rate
    print("\nbest solution")
    print_solution(best_solution, demands)
    

def read_initial_demand(file_path: str) -> tuple[int, int, DemandList]:
    
    demands = []
    
    with open(file_path, 'r') as f:
        start_node_id, end_node_id, num_demands = map(int, f.readline().strip().split())
        for i in range(num_demands):
            f.readline()
            capacity, start_time, end_time, num_pickup_choices = map(float, f.readline().strip().split())
            capacity = int(capacity)
            num_pickup_choices = int(num_pickup_choices)
            
            pickup_nodes = []
            for _ in range(num_pickup_choices):
                node_id = int(f.readline().strip())
                pickup_nodes.append(Node(node_id, i, "pickup", capacity))
            
            f.readline()
            
            num_delivery_choices = int(f.readline().strip())
            
            delivery_nodes = []
            for _ in range(num_delivery_choices):
                node_id = int(f.readline().strip())
                delivery_nodes.append(Node(node_id, i, "delivery", capacity))
            
            demand = Demand(pickup_nodes, delivery_nodes, capacity, start_time, end_time)
            demands.append(demand)
    
    return start_node_id, end_node_id, DemandList(demands)

def main():
    start_node_id, end_node_id, initial_demands = read_initial_demand("Data/initial_demand.txt")
    new_demands = DemandList([])
    ALNS(initial_demands, new_demands, max_iterations, start_node_id, end_node_id)
    
if __name__ == "__main__":
    main()


