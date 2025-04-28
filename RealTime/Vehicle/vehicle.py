from Node.node import Node
from Demands.demands import DemandList, Demand
from Parameters import vehicle_capacity, distance_matrix, Penalty_Coefficient,maximum_time_limit
import copy
import random

class Route:
    def __init__(self, nodes: list[Node], load: list[int], time: list[float]):
        self.nodes = nodes
        self.load = load
        self.time = time
    
    def validate_route(self):
        for i in range(len(self.nodes)):
            if self.load[i] > vehicle_capacity:
                return False
            if self.time[i] > maximum_time_limit:
                return False
        return True
class Vehicle:
    def __init__(self,index: int, start_node_id: int, end_node_id: int):
        self.capacity = vehicle_capacity
        self.route = Route(list[Node](), list[int](), list[float]())
        self.index = index
        self.demand_list = list[Demand]()
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        
    def get_start_node_id(self):
        return self.start_node_id

    def get_end_node_id(self):
        return self.end_node_id
        
    def get_capacity(self):
        return self.capacity
    
    def get_route(self):
        return self.route

    def append_node(self, node: Node):
        self.route.nodes.append(node)

    def remove_node(self, node: Node):
        if node in self.route.nodes:
            self.route.nodes.remove(node)
            
    def is_route_empty(self):
        return len(self.route.nodes) == 0
    
    def get_first_node(self):
        return self.route.nodes[0] if not self.is_route_empty() else None
    
    def get_last_node(self):
        return self.route.nodes[-1] if not self.is_route_empty() else None
    
    def get_load(self):
        return self.route.load
    
    def calculate_distance(self):
        if self.is_route_empty():
            return distance_matrix[self.get_start_node_id()][self.get_end_node_id()]
        else:
            distance = distance_matrix[self.get_start_node_id()][self.get_first_node().get_id()]
            for i in range(len(self.route.nodes) - 1):
                distance += distance_matrix[self.route.nodes[i].get_id()][self.route.nodes[i+1].get_id()]
            distance += distance_matrix[self.get_last_node().get_id()][self.get_end_node_id()]
            return distance
    
    def calculate_penalty(self, demandlist: DemandList):
        penalty = 0
        for i in range(len(self.route.nodes)):
            if self.route.time[i] > demandlist.get_demand(self.route.nodes[i].get_demand_index()).get_end_time():
                penalty += (self.route.time[i] - demandlist.get_demand(self.route.nodes[i].get_demand_index()).get_end_time()) * Penalty_Coefficient
        return penalty
    
    def calculate_total_cost(self, demandlist: DemandList):
        return self.calculate_distance() + self.calculate_penalty(demandlist)
    
    def get_index(self):
        return self.index

    def get_demand_list(self):
        return self.demand_list
    
    def append_demand(self, demand: Demand):
        self.demand_list.append(demand)

    def remove_demand(self, demand: Demand):
        for v_demand in self.demand_list:
            if v_demand.get_index() == demand.get_index():
                self.demand_list.remove(v_demand)
                break
    def remove_demand_nodes(self, demand: Demand, demandlist: DemandList):
        nodes_to_remove = []
        for node in self.route.nodes:
            if demandlist.get_demand_by_node(node).index == demand.get_index():
                nodes_to_remove.append(node)  
        for node in nodes_to_remove:
            self.route.nodes.remove(node)        
    
    def calculate_demand_cost(self, demand: Demand, demandlist: DemandList):
        current_cost = self.calculate_total_cost(demandlist)
        new_vehicle = copy.deepcopy(self)
        new_vehicle.remove_demand(demand)
        new_vehicle.remove_demand_nodes(demand, demandlist)
        new_vehicle.update_time(demandlist)
        new_cost = new_vehicle.calculate_total_cost(demandlist)
        return new_cost - current_cost

    def update_load(self):
        if self.is_route_empty():
            return []
        else:
            self.route.load = [0] * len(self.route.nodes)
            self.route.load[0] = self.route.nodes[0].get_demand_load()
            for i in range(len(self.route.nodes) - 1):
                current_node = self.route.nodes[i+1]
                current_load = self.route.load[i]
                if current_node.get_type() == "pickup":
                    self.route.load[i+1] = current_load + current_node.get_demand_load()
                elif current_node.get_type() == "delivery":
                    self.route.load[i+1] = current_load - current_node.get_demand_load()
    
    def update_time(self, demandlist: DemandList):
        if self.is_route_empty():
            return []
        else:
            self.route.time = [0] * len(self.route.nodes)
            self.route.time[0] = distance_matrix[self.get_start_node_id()][self.get_first_node().get_id()]
            for i in range(len(self.route.nodes) - 1):
                current_time = self.route.time[i]
                current_node = self.route.nodes[i+1]
                current_time += distance_matrix[current_node.get_id()][self.route.nodes[i].get_id()]
                if current_time < demandlist.get_demand(current_node.get_demand_index()).get_start_time():
                    current_time = demandlist.get_demand(current_node.get_demand_index()).get_start_time()
                self.route.time[i+1] = current_time
    
    def insert_demand(self, demand: Demand, demandlist: DemandList):
        best_cost = float('inf')
        best_status = None
        for i in range(len(self.route.nodes)):
            for pickup_node in demand.get_pickup_nodes():
                for j in range(i, len(self.route.nodes)):
                    for delivery_node in demand.get_delivery_nodes():
                        temp_vehicle = copy.deepcopy(self)
                        temp_vehicle.route.nodes.insert(i, pickup_node)
                        temp_vehicle.route.nodes.insert(j+1, delivery_node)
                        temp_vehicle.update_load()
                        temp_vehicle.update_time(demandlist)
                        temp_vehicle.append_demand(demand)
                        if temp_vehicle.route.validate_route():
                            temp_cost = temp_vehicle.calculate_total_cost(demandlist)
                            if temp_cost < best_cost:
                                best_cost = temp_cost
                                best_status = temp_vehicle
        return best_status, best_cost 
    
    def random_insert(self, demand: Demand,demandlist: DemandList):
        pickup_index = random.randint(0, len(self.route.nodes))
        pickup_node = random.choice(demand.get_pickup_nodes())
        delivery_index = random.randint(pickup_index, len(self.route.nodes))
        delivery_node = random.choice(demand.get_delivery_nodes())
        new_vehicle = copy.deepcopy(self)
        new_vehicle.route.nodes.insert(pickup_index, pickup_node)
        new_vehicle.route.nodes.insert(delivery_index+1, delivery_node)
        new_vehicle.update_load()
        new_vehicle.update_time(demandlist)
        new_vehicle.append_demand(demand)
        if new_vehicle.route.validate_route():
            return new_vehicle
        else:
            return None
