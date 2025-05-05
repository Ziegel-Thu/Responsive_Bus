from Demands.demands import DemandList, Demand
from Node.node import Node
from Parameters2 import vehicle_capacity, distance_matrix, initial_temperature, cooling_rate, max_iterations, max_recent_costs, identical_solution_threshold
import math
import random
import copy
from Vehicle.vehicle import Vehicle
import time
import sys
import os
import matplotlib

class VRPPD:
    def __init__(self, demands: DemandList, start_node_id: int, end_node_id: int, num_vehicles: int):
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
                    min_vehicle.append_node(min_node)
                    picked_demands[min_vehicle].append(min_demand)
                    remaining_demands.remove(min_demand)
                min_vehicle.update_load()
                vehicle_clock[min_vehicle] += min_distance
            else:
                Done = True
        for vehicle in self.vehicle_list:
            vehicle.update_time(self.demands)
        print("Basic case generated\n")
        return self.vehicle_list
    
def destroy_demands(vehicle_list: list[Vehicle], destroyed_demands: list[Demand], demandlist: DemandList, optional_demands: list[Demand], available_demands: list[Demand]):
    for vehicle in vehicle_list:
        nodes_to_remove = []
        demands_to_remove : list[Demand] = []

        for node in vehicle.get_route().nodes:
            if demandlist.get_demand_by_node(node) in destroyed_demands:
                if node.get_type() == "delivery":
                    demands_to_remove.append(demandlist.get_demand_by_node(node))
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            vehicle.remove_node(node)
        for demand in demands_to_remove:
            vehicle.remove_demand(demand)
            optional_demands.remove(demand)
            available_demands.append(demand)

        vehicle.update_load()
        vehicle.update_time(demandlist)

def random_destroy(vehicle_list: list[Vehicle], destroyable_demands: list[Demand], demands:DemandList, destruction_rate: float, available_demands : list[Demand]):
    destroy_num = math.ceil(len(destroyable_demands) * destruction_rate)
    destroyed_demands = random.sample(destroyable_demands, destroy_num)
    destroy_demands(vehicle_list, destroyed_demands,demands, destroyable_demands, available_demands)
    
def find_worst_demand(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: list[Demand]):
    worst_demand = None
    worst_cost = float('inf')
    for demand in destroyable_demands:
        flag = False
        for vehicle in vehicle_list:
            for v_demand in vehicle.get_demand_list():
                if v_demand.index == demand.index :
                    cost = vehicle.calculate_demand_cost(demand, demandlist)
                    if cost < worst_cost:
                        worst_cost = cost
                        worst_demand = demand
                    flag = True
                    break
            if flag:
                break
    return worst_demand

def find_bad_cost(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: list[Demand]):
    bad_costs = {}  # Using dictionary to map demands to costs
    for demand in destroyable_demands:
        flag = False
        for vehicle in vehicle_list:
            for v_demand in vehicle.get_demand_list():
                if v_demand.index == demand.index:
                    cost = vehicle.calculate_demand_cost(demand, demandlist)
                    if cost != 0:
                        bad_costs[demand.index] = cost  # Map demand to its cost
                        flag = True
                        break
            if flag:
                break
    return bad_costs

def find_bad_demand(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: list[Demand]):
    cost_list  = find_bad_cost(vehicle_list, demandlist, destroyable_demands)
    avg_cost = sum(cost_list.values()) / len(cost_list)
    normalized_costs = {demand_id: cost for demand_id, cost in cost_list.items()} if avg_cost else {}
    # Calculate probabilities using softmax
    exp_costs = {demand_id: cost for demand_id, cost in normalized_costs.items()}
    total_exp = sum(exp_costs.values())
    probabilities = {demand_id: exp/total_exp for demand_id, exp in exp_costs.items()}

    # Select a demand based on probabilities
    rand = random.random()
    cumulative_prob = 0
    bad_demand = None
    for demand_id, prob in probabilities.items():
        cumulative_prob += prob
        if rand <= cumulative_prob:
            bad_demand = next((d for d in destroyable_demands if d.index == demand_id), None)
            break

    return bad_demand

def bad_destroy(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: list[Demand], destruction_rate: float, available_demands: list[Demand]):
    destroy_num = math.ceil(len(destroyable_demands) * destruction_rate)
    destroyed_demands = []
    for _ in range(destroy_num):
        bad_demand = find_bad_demand(vehicle_list, demandlist, destroyable_demands)
        print(bad_demand.get_index())
        if bad_demand:
            destroyed_demands.append(bad_demand)
            destroy_demands(vehicle_list, [bad_demand], demandlist, destroyable_demands, available_demands )
    return destroyed_demands
        
def worst_destroy(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: list[Demand], destruction_rate: float, available_demands: list [Demand]):
    destroy_num = math.ceil(len(destroyable_demands) * destruction_rate)
    destroyed_demands = []
    for _ in range(destroy_num):
        worst_demand = find_worst_demand(vehicle_list, demandlist, destroyable_demands)
        if worst_demand:
            destroyed_demands.append(worst_demand)
            destroy_demands(vehicle_list, [worst_demand], demandlist, destroyable_demands,available_demands )
            
    return destroyed_demands
    
    
def best_repair(vehicle_list: list[Vehicle], demandlist: DemandList, repairing_demands: list[Demand],optional_demands: list[Demand]):
    new_vehicle_list = copy.deepcopy(vehicle_list)
    possible = True
    while possible:
        best_choice = None
        best_change = None
        best_demand = None
        best_cost = float('inf')
        for demand in repairing_demands:
            for vehicle_id in range(num_vehicles):
                best_update, cost = new_vehicle_list[vehicle_id].insert_demand(demand,demandlist)
                if best_update:
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = vehicle_id
                        best_change = best_update
                        best_demand = demand
        if best_change:
            optional_demands.append(best_demand)
            repairing_demands.remove(best_demand)
            new_vehicle_list[best_choice] = best_change
            print("repaired", best_demand.index)
            for demand in repairing_demands:
                print(demand.index)
        else:
            possible = False
    return new_vehicle_list

def random_repair(vehicle_list: list[Vehicle], repairing_demands: list[Demand],demandlist: DemandList, optional_demands: list[Demand]):
    new_vehicle_list = copy.deepcopy(vehicle_list)
    failure_count = 0
    max_failure = math.ceil(len(repairing_demands) * 0.3)
    success = False
    while failure_count < max_failure and repairing_demands:
        demand = random.choice(repairing_demands)
        vehicle_id = random.choice(range(num_vehicles))
        new_vehicle = new_vehicle_list[vehicle_id].random_insert(demand,demandlist)
        if new_vehicle:
            new_vehicle_list[vehicle_id] = new_vehicle
            repairing_demands.remove(demand)
            optional_demands.append(demand)
            success = True
        else:
            failure_count += 1
    return new_vehicle_list, success

def calculate_cost(vehicle_list: list[Vehicle], demands: DemandList):
    load_count = 0
    total_cost = 0
    for vehicle in vehicle_list:
        total_cost += vehicle.calculate_total_cost(demands)
        for demand in vehicle.get_demand_list():
            load_count += demand.demand_load
    return total_cost, load_count

def print_solution(vehicle_list: list[Vehicle], demands: DemandList):
    for vehicle in vehicle_list:
        print("vehicle ",vehicle.get_index())
        i = 0
        for node in vehicle.get_route().nodes:
            print(node.get_demand_index()," ",node.get_id()," ",node.get_type()," ",node.get_demand_load()," ",vehicle.route.time[i])
            i +=1
        vehicle_cost = vehicle.calculate_total_cost(demands)
        print("costs: ",vehicle_cost,"\n")
    total_cost, load_serviced= calculate_cost(vehicle_list, demands)
    print("total cost: ",total_cost, "load_serviced", load_serviced,"\n")
    
def ALNS(initial_demands: DemandList, new_demands: DemandList, max_iterations: int, start_node: int, end_node: int,num_vehicles: int):
    vrppd = VRPPD(initial_demands, start_node, end_node, num_vehicles)
    vehicle_list = vrppd.vehicle_list
    print_solution(vehicle_list, initial_demands)
    demands = copy.deepcopy(initial_demands)
    for demand in new_demands.get_all_demands():
        demands.add_demand(demand)
    available_demands = new_demands.get_all_demands()
    optional_demands = []
    new_vehicle_list, success = random_repair(vehicle_list, available_demands, demands,optional_demands)
    if new_vehicle_list:
        vehicle_list = new_vehicle_list
    else:
        print("invalid data!")
    print("\nInitial complemented solution generated")
    print_solution(vehicle_list,demands)
    time_start = time.time()
    iteration = 0
    recent_costs= []
    need_random_destroy = False
    temporary_best_solution = vehicle_list
    temporary_best_cost, temporary_most_load = calculate_cost(temporary_best_solution, demands)
    print(temporary_best_cost, temporary_most_load)
    destroy_rate = 0.2
    temperature = initial_temperature
    best_cost = temporary_best_cost
    best_solution = temporary_best_solution
    most_load = temporary_most_load
    current_optional_demands = []
    current_available_demands = []
    load = []
    cost = []
    load.append(temporary_most_load)
    cost.append(temporary_best_cost)
    while iteration < max_iterations:
        print("\nIteration: ",iteration)
        if need_random_destroy:
            #print("processing random destroy")
            random_destroy(current_solution, optional_demands,demands, destroy_rate,available_demands)
            need_random_destroy = False
            new_vehicle_list, success = random_repair(current_solution, available_demands, demands, optional_demands)
            iteration +=1
            temperature *= cooling_rate
            if success:
                #print("sucessfully repaired")
                vehicle_list = new_vehicle_list
                #print("random result force accepted")
                recent_costs.clear()
            else:
                vehicle_list = current_solution
                #print_solution(vehicle_list, demands)
                recent_costs.clear()
            random_cost, random_load_serviced = calculate_cost (vehicle_list, demands)
            temporary_best_cost = random_cost
            temporary_most_load = random_load_serviced
            load.append(temporary_most_load)
            cost.append(temporary_best_cost)
        else:
            
            current_solution = copy.deepcopy(vehicle_list)
            current_optional_demands = copy.deepcopy(optional_demands)
            current_available_demands = copy.deepcopy(available_demands)
            print("processing bad destroy")
            for demand in optional_demands:
                    print(demand.index)
            print_solution(current_solution,demands)

            bad_destroy(current_solution, demands,optional_demands, destroy_rate,available_demands)
            print("after destroy")
            print_solution(current_solution,demands)
            #print("processing best rrrrepair")
            for demand in optional_demands:
                print(demand.get_index())
            print('____')
            for demand in available_demands:
                print(demand.get_index())
            new_vehicle_list = best_repair(current_solution,demands, available_demands, optional_demands)
            
            if new_vehicle_list:
                print("repaired")
                #print_solution(new_vehicle_list,demands)
                current_solution = new_vehicle_list
                print_solution(current_solution,demands)
                repairable = True
            while not repairable:
                print("best repair failed, processing random repair")
                new_vehicle_list, success = random_repair(vehicle_list, available_demands, demands, optional_demands)
                if success:
                    vehicle_list = new_vehicle_list
                else:
                    repairable = False
        
            iteration += 1
            current_cost, load_serviced = calculate_cost(current_solution, demands)
            cost_diff = current_cost - temporary_best_cost
            if(cost_diff < 0 and load_serviced == most_load )or (load_serviced > temporary_most_load)or random.random() < math.exp(-cost_diff -   (most_load - load_serviced)/ temperature):
                vehicle_list = current_solution
                temporary_best_cost = current_cost
                temporary_most_load = load_serviced
                #print("\nnew solution accepted")
                recent_costs.append(current_cost)
                if len(recent_costs) > max_recent_costs:
                    recent_costs.pop(0)
                if recent_costs.count(current_cost) > identical_solution_threshold:
                    need_random_destroy = True
                #print_solution(vehicle_list, demands)
            else:
                print("\nnew solution rejected")
                optional_demands = copy.deepcopy(current_optional_demands)
                available_demands = copy.deepcopy(current_available_demands)
                for demand in optional_demands:
                    print(demand.index)
            load.append(temporary_most_load)
            cost.append(temporary_best_cost)
            if (temporary_best_cost < best_cost and temporary_most_load == most_load) or most_load < temporary_most_load:
                best_cost = temporary_best_cost
                best_solution = vehicle_list
                most_load = temporary_most_load
            temperature *= cooling_rate
        sum =0
        for vehicle in vehicle_list:
            sum += len(vehicle.route.nodes)
            if len(vehicle.route.nodes)!= 2*len(vehicle.demand_list):
                sys.exit()
        if(sum != 2* (initial_demands.get_num_demands()+len(optional_demands))):
            sys.exit()
        
        print_solution(vehicle_list, demands)
        print(initial_demands.get_num_demands()+len(optional_demands))
        print(sum)
    print("\nbest solution")
    print_solution(best_solution, demands)

    
    return best_cost, most_load, time.time()-time_start
    

def read_initial_demand(file_path: str) -> tuple[int, int, DemandList]:
    
    demands = []
    
    with open(file_path, 'r') as f:
        start_node_id, end_node_id, num_demands = map(int, f.readline().strip().split())
        for i in range(num_demands):
            f.readline()
            capacity, start_time, end_time = map(float, f.readline().strip().split())
            capacity = int(capacity)
            
            pickup_nodes = []
            line = f.readline().strip()
            numbers = list(map(int, line.split()))
            for node_id in numbers:
                pickup_nodes.append(Node(node_id, i, "pickup", capacity))                
            
            f.readline()
            
            
            delivery_nodes = []
            line = f.readline().strip()
            numbers = list(map(int, line.split()))
            for node_id in numbers:
                delivery_nodes.append(Node(node_id, i, "delivery", capacity))
            
            demand = Demand(pickup_nodes, delivery_nodes, capacity, start_time, end_time, i)
            demands.append(demand)
    
    return start_node_id, end_node_id, DemandList(demands)

def read_new_demand(file_path: str, num_initial_demands:int) -> DemandList:
    demands = []
    with open(file_path, 'r') as f:
        num_demands = int(f.readline().strip())
        for i in range(num_demands):
            f.readline()
            capacity, start_time, end_time = map(float, f.readline().strip().split())
            capacity = int(capacity)
            
            pickup_nodes = []
            line = f.readline().strip()
            numbers = list(map(int, line.split()))
            for node_id in numbers:
                pickup_nodes.append(Node(node_id, i+ num_initial_demands, "pickup", capacity))                
            
            f.readline()
            
            
            delivery_nodes = []
            line = f.readline().strip()
            numbers = list(map(int, line.split()))
            for node_id in numbers:
                delivery_nodes.append(Node(node_id, i+ num_initial_demands, "delivery", capacity))
            
            demand = Demand(pickup_nodes, delivery_nodes, capacity, start_time, end_time, i+ num_initial_demands)
            demands.append(demand)
    return DemandList(demands)

def main():
    if not os.path.exists("Data/Result_up2"):
        os.makedirs("Data/Result_up2")
    for k in [5]:
        for a in [6]:
            for b in [0,0.5,1]:
                output_file = f"Data/Result_up2/result_{k}_{a}_{b}.txt"
                global num_vehicles
                num_vehicles = k
                start_node_id, end_node_id, initial_demands = read_initial_demand(f"Data/demand/output_{k}_{a}_{b}_init.txt")
                num_initial_demands = initial_demands.get_num_demands()
                new_demands = read_new_demand(f"Data/demand/output_{k}_{a}_{b}_new.txt", num_initial_demands)
                with open(output_file, 'w') as f:
                    original_stdout = sys.stdout
                    sys.stdout = f
                    cost,load,time =ALNS(initial_demands, new_demands, max_iterations, start_node_id, end_node_id, num_vehicles)
                    sys.stdout = original_stdout
                print(k,a,b,cost,load,time)      
if __name__ == "__main__":
    main()


