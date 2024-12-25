class Node:
    def __init__(self, id: int, demand_index: int, type: str, demand_load: int):
        self.id = id
        self.demand_index = demand_index
        self.type = type
        self.demand_load = demand_load

    def get_id(self):
        return self.id
    
    def get_demand_index(self):
        return self.demand_index
    
    def get_type(self):
        return self.type
    
    def get_demand_load(self):
        return self.demand_load

        