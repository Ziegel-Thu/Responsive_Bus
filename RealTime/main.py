from Demands.demands import DemandList, Demand
from Node.node import Node
from Parameters import vehicle_capacity, distance_matrix, initial_temperature, cooling_rate, max_iterations, max_recent_costs, identical_solution_threshold
import math
import random
import copy
from Vehicle.vehicle import Vehicle
import time
import sys
import os
import pandas as pd
import numpy as np
from gurobipy import *
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
                print("destroyed", node.demand_index)
                if node.get_type() == "delivery":
                    demands_to_remove.append(demandlist.get_demand_by_node(node))
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            vehicle.remove_node(node)
        for demand in demands_to_remove:
            print("\nremoving demand with index", demand.get_index())

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
                print(v_demand.index)
                if v_demand.index == demand.index :
                    print("\ndemand",demand.index)
                    print("vehicle", vehicle.get_index())
                    cost = vehicle.calculate_demand_cost(demand, demandlist)
                    print(cost)
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
                    else:
                        print("no change")
            if flag:
                break
    return bad_costs

def find_bad_demand(vehicle_list: list[Vehicle], demandlist: DemandList, destroyable_demands: list[Demand]):
    cost_list  = find_bad_cost(vehicle_list, demandlist, destroyable_demands)
    print("find bad demand with in")
    for temp_demand in destroyable_demands:
        print(temp_demand.index)
    print(cost_list)
    avg_cost = sum(cost_list.values()) / len(cost_list)
    normalized_costs = {demand_id: cost/avg_cost for demand_id, cost in cost_list.items()} if avg_cost else {}
    # Calculate probabilities using softmax
    exp_costs = {demand_id: math.exp(cost) for demand_id, cost in normalized_costs.items()}
    total_exp = sum(exp_costs.values())
    probabilities = {demand_id: exp/total_exp for demand_id, exp in exp_costs.items()}

    # Select a demand based on probabilities
    rand = random.random()
    cumulative_prob = 0
    bad_demand = None
    print(probabilities)
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
                    print(cost,demand.index,vehicle_id)
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = vehicle_id
                        best_change = best_update
                        best_demand = demand
                else:
                    print("repair not found")        
        if best_change:
            optional_demands.append(best_demand)
            repairing_demands.remove(best_demand)
            new_vehicle_list[best_choice] = best_change
            print("success")
            for node in best_change.get_route().nodes:
                print(demandlist.get_demand_by_node(node).index," ", best_change.get_index())
        else:
            possible = False
    return new_vehicle_list

def random_repair(vehicle_list: list[Vehicle], repairing_demands: list[Demand],demandlist: DemandList, optional_demands: list[Demand]):
    new_vehicle_list = copy.deepcopy(vehicle_list)
    failure_count = 0
    max_failure = math.ceil(len(repairing_demands) * 0.3)
    success = False
    print("\navailable demands")
    for demand in repairing_demands:
        print(demand.index)
    while failure_count < max_failure and repairing_demands:
        demand = random.choice(repairing_demands)
        vehicle_id = random.choice(range(num_vehicles))
        new_vehicle = new_vehicle_list[vehicle_id].random_insert(demand,demandlist)
        if new_vehicle:
            print("repaired",demand.index,vehicle_id)
            new_vehicle_list[vehicle_id] = new_vehicle
            repairing_demands.remove(demand)
            optional_demands.append(demand)
            success = True
            print(success)
        else:
            failure_count += 1
            print("fail")
    print("\noptional demands")
    for demand in optional_demands:
        print(demand.index)
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
        for node in vehicle.get_route().nodes:
            print(node.get_demand_index()," ",node.get_id()," ",node.get_type()," ",node.get_demand_load())
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
    while iteration < max_iterations:
        print("\nIteration: ",iteration)
        if need_random_destroy:
            print("processing random destroy")
            print("\n before destroy")
            print_solution(current_solution, demands)
            random_destroy(current_solution, optional_demands,demands, destroy_rate,available_demands)
            print("\nafter destroy!")
            print_solution(current_solution, demands)
            need_random_destroy = False
            print("processing random repair")
            new_vehicle_list, success = random_repair(current_solution, available_demands, demands, optional_demands)
            iteration +=1
            temperature *= cooling_rate
            if success:
                print("sucessfully repaired")
                vehicle_list = new_vehicle_list
                print("random result force accepted")
                recent_costs.clear()
            else:
                vehicle_list = current_solution
                print_solution(vehicle_list, demands)
                recent_costs.clear()
            random_cost, random_load_serviced = calculate_cost (vehicle_list, demands)
            temporary_best_cost = random_cost
            temporary_most_load = random_load_serviced

        else:
            
            current_solution = copy.deepcopy(vehicle_list)
            current_optional_demands = optional_demands
            current_available_demands = available_demands
            print("processing bad destroy")
            bad_destroy(current_solution, demands,optional_demands, destroy_rate,available_demands)
            print_solution(current_solution, demands)
            print("processing best rrrrepair")
            print("num:",len(available_demands))
            for demand in available_demands:
                print(demand.index)
            new_vehicle_list = best_repair(current_solution,demands, available_demands, optional_demands)
            
            if new_vehicle_list:
                print("repaired")
                print_solution(new_vehicle_list,demands)
                current_solution = new_vehicle_list
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
                print("\nnew solution accepted")
                recent_costs.append(current_cost)
                if len(recent_costs) > max_recent_costs:
                    recent_costs.pop(0)
                if recent_costs.count(current_cost) > identical_solution_threshold:
                    need_random_destroy = True
                print_solution(vehicle_list, demands)
            else:
                print("\nnew solution rejected")
                optional_demands = current_optional_demands
                available_demands = current_available_demands
                print(len(optional_demands))
                print(len(available_demands))
            if (temporary_best_cost < best_cost and temporary_most_load == most_load) or most_load < temporary_most_load:
                best_cost = temporary_best_cost
                best_solution = vehicle_list
                most_load = temporary_most_load
            temperature *= cooling_rate
        sum =0
        for vehicle in vehicle_list:
            sum += len(vehicle.route.nodes)
        if sum !=(2*len(optional_demands) +2* (initial_demands.get_num_demands())):
            print_solution(vehicle_list, demands)
            for t_demand in optional_demands:
                print(t_demand.index)
            sys.exit("error")
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
    if not os.path.exists("Data/Result"):
        os.makedirs("Data/Result")

    for k in [1,3,5]:
        for a in [3,6,9]:
            for b in [0,0.5,1]:
                output_file = f"Data/Result/result_{k}_{a}_{b}.txt"
                global num_vehicles
                num_vehicles = k
                start_node_id, end_node_id, initial_demands = read_initial_demand(f"Data/demand/output_{k}_{a}_{b}_init.txt")
                num_initial_demands = initial_demands.get_num_demands()
                new_demands = read_new_demand(f"Data/demand/output_{k}_{a}_{b}_new.txt", num_initial_demands)
                with open(output_file, 'w') as f:
                    original_stdout = sys.stdout
                    sys.stdout = f
                    cost,load,time =ALNS(initial_demands, new_demands, max_iterations, start_node_id, end_node_id, num_vehicles )
                    sys.stdout = original_stdout
                    print(k,a,b,cost,load,time)

class GurobiModel():
    def __init__(self,k,a,b):
        self.model = None
        self.vehicle_number = k
        self.crowdness = a
        self.candidate = b
        
    def load_initial_data(self):
        self.start_node_id, self.end_node_id, self.initial_demands = read_initial_demand(f"Data/demand/output_{self.vehicle_number}_{self.crowdness}_{self.candidate}_init.txt")
        self.new_demands = read_new_demand(f"Data/demand/output_{self.vehicle_number}_{self.crowdness}_{self.candidate}_new.txt", self.initial_demands.get_num_demands())
        self.vrppd = VRPPD(self.initial_demands, self.start_node_id, self.end_node_id, self.vehicle_number)
        self.initial_solution = self.vrppd.vehicle_list
        self.start_node = Node(self.start_node_id, -1, "start", 0)
        self.end_node = Node(self.end_node_id, -1, "end", 0)

    
    def point_mapping(self):
        count = 0
        num_to_point: list[Node] = []
        point_to_num = {}
        for demand in self.initial_demands.get_all_demands():
            for node in demand.get_pickup_nodes():
                num_to_point.append(node)
                point_to_num[node.get_id()] = count
                count += 1
            for node in demand.get_delivery_nodes():
                num_to_point.append(node)
                point_to_num[node.get_id()] = count
                count += 1
        for demand in self.new_demands.get_all_demands():
            for node in demand.get_pickup_nodes():
                num_to_point.append(node)
                point_to_num[node.get_id()] = count
                count += 1
            for node in demand.get_delivery_nodes():
                num_to_point.append(node)
                point_to_num[node.get_id()] = count
                count += 1
        num_to_point.append(self.start_node)
        point_to_num[self.start_node] = count
        count += 1
        num_to_point.append(self.end_node)
        point_to_num[self.end_node] = count      
        self.num_to_point = num_to_point
        self.point_to_num = point_to_num
        self.num_customers = self.new_demands.get_num_demands() + self.initial_demands.get_num_demands()
        
        
    def build_model(self):
        demands = copy.deepcopy(self.initial_demands.get_all_demands())
        for demand in self.new_demands.get_all_demands():
            demands.append(demand)
        distance_data = pd.read_csv('Data/wangjing-newdis-expanded.csv').values
        self.model = Model()
        N = len(self.num_to_point)
        M = self.num_customers
        gurobi_distance = self.model.addVars (N, N+2, self.vehicle_number, vtype=GRB.BINARY, name="gurobi_distance")
        gurobi_time = self.model.addVars(2 * M, vtype=GRB.CONTINUOUS, name="gurobi_time")
        gurobi_overtime  = self.model.addVars(M, vtype=GRB.CONTINUOUS, name="gurobi_overtime")
        gurobi_load = self.model.addVars(2 * M, vtype=GRB.CONTINUOUS, name="gurobi_load")
        obj = LinExpr(0)
        for i in range(N ):
            for j in range(N):
                for k in range(self.vehicle_number):
                    if i != j:
                        obj += gurobi_distance[i,j,k] * distance_data[self.num_to_point[i].get_id(), self.num_to_point[j].get_id()]
        for j in range(M):
            obj += 3 * gurobi_overtime[j]
        
        
        self.model.setObjective(obj, GRB.MINIMIZE)

        for vehicle in self.initial_solution:
            for demand in vehicle.get_demand_list():
                expr1 = LinExpr(0)
                expr2 = LinExpr(0)
                for node in demand.get_pickup_nodes():
                    for j in range(N):
                        if node.get_id() != self.num_to_point[j].get_id():
                            expr1.addTerms(1, gurobi_distance[self.point_to_num[node.get_id()], j, vehicle.get_index()])
                for node in demand.get_delivery_nodes():
                    for j in range(N):
                        if node.get_id() != self.num_to_point[j].get_id():
                            expr2.addTerms(1, gurobi_distance[self.point_to_num[node.get_id()], j, vehicle.get_index()])
                self.model.addConstr(expr1 == 1, name=f'init_pickup_{demand.get_index()}')
                self.model.addConstr(expr2 == 1, name=f'init_delivery_{demand.get_index()}')
                
        for demand in self.new_demands.get_all_demands():
            expr3 = LinExpr(0)
            expr4 = LinExpr(0)
            for node in demand.get_pickup_nodes():
                for j in range(N):
                    if node.get_id() != self.num_to_point[j].get_id():
                        for k in range (self.vehicle_number):
                            expr3.addTerms(1, gurobi_distance[self.point_to_num[node.get_id()], j, vehicle.get_index()])
            for node in demand.get_delivery_nodes():
                for j in range(N):
                    if node.get_id() != self.num_to_point[j].get_id():
                        for k in range (self.vehicle_number):
                            expr4.addTerms(1, gurobi_distance[self.point_to_num[node.get_id()], j, vehicle.get_index()])
            self.model.addConstr(expr3 == 1, name=f'new_pickup_{demand.get_index()}')
            self.model.addConstr(expr4 == 1, name=f'new_delivery_{demand.get_index()}')
        
        for k in range(self.vehicle_number):
            expr5 = LinExpr(0)
            for j in range(N-2):
                expr5.addTerms(1, gurobi_distance[j, N-1, k])
                self.model.addConstr((gurobi_distance [j, N-1, k]==1)>>(6 - gurobi_time[self.num_to_point[j].get_demand_index() + M * int(self. num_to_point[j].get_type() == "delivery")] - distance_matrix[self.end_node.get_id()][self.num_to_point[j].get_id()] >=0), name = f'last_point_{j}')
            self.model.addConstr(expr5 == 1, name=f'cons_end_{k}')
            
        for k in range(self.vehicle_number):
            expr6 = LinExpr(0)
            for j in range(N-2):
                expr6.addTerms(1, gurobi_distance[N-2, j, k])
                self.model.addConstr((gurobi_distance[N-2, j, k]==1)>>((gurobi_time[self.num_to_point[j].get_demand_index() + M * int(self. num_to_point[j].get_type() == "delivery")] - distance_matrix[self.start_node_id][self.num_to_point[j].get_id()]) >=0), name = f'first_point_{j}' )
                self.model.addConstr((gurobi_distance[N-2, j, k]==1)>>(gurobi_load[self.num_to_point[j].get_demand_index() + M * int(self. num_to_point[j].get_type() == "delivery")] - demands[self.num_to_point[j].get_demand_index()].demand_load* (2* (self. num_to_point[j].get_type() == "delivery") - 1)== 0),name = f'first_load_{j}')
            expr6.addTerms(1,gurobi_distance[N-2, N-1, k])
            self.model.addConstr(expr6 == 1, name=f'cons_start_{k}')
        
        for k in range(self.vehicle_number):
            for i in range(0, N-2):
                expr7 = LinExpr(0)
                for j in range (N):
                    if i != j:
                        expr7.addTerms(1, gurobi_distance[i, j, k])
                        expr7.addTerms(-1, gurobi_distance[j, i, k])
                self.model.addConstr(expr7 == 0, name=f'cons_in_and_out_{i}_{k}')
        
        for i in range (N-2):
            for j in range (N-2):
                for k in range (self.vehicle_number):
                    if i !=j :
                        self.model.addConstr((gurobi_distance[i,j,k] ==1)>>((gurobi_time[self.num_to_point[j].get_demand_index() + M * int(self.num_to_point[j].get_type() == "delivery") ] - gurobi_time[self.num_to_point[i].get_demand_index() + M * (self.num_to_point[i].get_type() == "delivery") ] - distance_matrix[self.num_to_point[i].get_id()][self.num_to_point[j].get_id()]) >= 0), name = f'distance_{i}_{j}_{k}')
                
        for l in range (M):
            self.model.addConstr(gurobi_time[l] <= gurobi_time[l+M], name = f'order_{l}' )
        for l in range (M):
            self.model.addConstr(gurobi_time[l] >= demands[l].get_start_time(),name = f'start_window_{l}')
        for l in range (M):
            self.model.addConstr(gurobi_time[l+M] - demands[l].get_end_time() <= gurobi_overtime[l], name = f'end_window_{l}')
        for l in range (M):
            self.model.addConstr(gurobi_overtime[l] >= 0, name = f'penalty_{l}')
        
        for i in range(N-2):
            for j in range (N-2):
                if i != j:
                    for k in range (self.vehicle_number):
                        self.model.addConstr((gurobi_distance[i,j,k]==1)>>(gurobi_load[self.num_to_point[j].get_demand_index() + M * int(self. num_to_point[j].get_type() == "delivery")] + demands[self.num_to_point[i].get_demand_index()].demand_load* (2* (self. num_to_point[j].get_type() == "delivery") - 1)== gurobi_load[self.num_to_point[j].get_demand_index() + M * (self. num_to_point[j].get_type() == "delivery")] ),name  = f'load_{i}_{j}_{k}')

        for l in range (2* M):
            self.model.addConstr(gurobi_load[l] <= 100, name = f'max_load{l}')
    def optimize(self):
        self.model.optimize()       
def gurobimain():
    for k in [1]:
        for a in [3]:
            for b in [0]:
                model = GurobiModel(k,a,b)

                model.load_initial_data()
                model.point_mapping()
                model.build_model()
                model.optimize()


    
if __name__ == "__main__":
    gurobimain()


