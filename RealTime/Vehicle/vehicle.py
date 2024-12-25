from Node.node import Node
from Demands.demands import DemandList
from Parameters import vehicle_capacity, distance_matrix, Penalty_Coefficient

class Vehicle:
    def __init__(self,index: int):
        self.capacity = vehicle_capacity
        self.route = list[Node]()
        self.current_location = Node(0, 0, "")
        self.load = 0
        self.index = index
        
    def get_capacity(self):
        return self.capacity
    
    def get_route(self):
        return self.route
    
    def get_current_location(self):
        return self.current_location
    
    def set_current_location(self, node: Node):
        self.current_location = node
    
    #def set_route(self, route: Route):
    #    self.route = route

    def append_node(self, node: Node):
        self.route.append_node(node)

    def remove_node(self, node: Node):
        if node in self.route:
            self.route.remove(node)
            
    def is_route_empty(self):
        return len(self.route) == 0
    
    def get_first_node(self):
        return self.route[0] if not self.is_route_empty() else None
    
    def get_last_node(self):
        return self.route[-1] if not self.is_route_empty() else None

    def update_load(self, node: Node):
        if node.get_type() == "pickup":
            self.load += node.get_demand_load()
        elif node.get_type() == "delivery":
            self.load -= node.get_demand_load()
    
    def get_load(self):
        return self.load
    
    def calculate_distance(self):
        distance = distance_matrix[0][self.get_first_node()]
        for i in range(len(self.route) - 1):
            distance += distance_matrix.loc[self.route[i].get_id(), self.route[i+1].get_id()]
        distance += distance_matrix[self.get_last_node()][0]
        return distance
    
    def calculate_penalty(self, demandlist: DemandList):
        penalty = 0
        current_time = distance_matrix[0][self.get_first_node()]
        for i in range(len(self.route) - 1):
            current_node = self.route[i]
            next_node = self.route[i+1]
            if current_node.get_type() == "pickup":
                if current_time < demandlist.get_demand(current_node.get_id()).get_start_time():
                    current_time = demandlist.get_demand(current_node.get_id()).get_start_time()
            elif current_node.get_type() == "delivery":
                if current_time > demandlist.get_demand(current_node.get_id()).get_end_time():
                    penalty += (current_time - demandlist.get_demand(current_node.get_id()).get_end_time()) * Penalty_Coefficient
            current_time += distance_matrix.loc[current_node.get_id(), next_node.get_id()]
        return penalty

    def calculate_total_cost(self, demandlist: DemandList):
        return self.calculate_distance() + self.calculate_penalty(demandlist)
    
    def get_index(self):
        return self.index
