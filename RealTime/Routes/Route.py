from Node.node import Node
from Demands.demands import DemandList
from Parameters import Penalty_Coefficient, distance_matrix

class Route:
    def __init__(self):
        self.route: list[Node] = []
        self.load = 0

    def append_node(self, node: Node):
        self.route.append(node)

    def remove_node(self, node: Node):
        if node in self.route:
            self.route.remove(node)

    def get_route(self):
        return self.route

    def get_load(self):
        return self.load

    def update_load(self, node: Node):
        if node.get_type() == "pickup":
            self.load += node.get_demand_load()
        elif node.get_type() == "delivery":
            self.load -= node.get_demand_load()

    def is_empty(self):
        return len(self.route) == 0
    
    def get_first_node(self):
        return self.route[0] if not self.is_empty() else None

    def get_last_node(self):
        return self.route[-1] if not self.is_empty() else None

    def calculate_distance(self):
        distance = 0
        for i in range(len(self.route) - 1):
            distance += distance_matrix.loc[self.route[i].get_id(), self.route[i+1].get_id()]
        return distance
    
    def calculate_penalty(self, demandlist: DemandList):
        penalty = 0
        current_time = distance_matrix.loc[0, self.route[0].get_id()]
        for i in range(len(self.route)-1):
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

