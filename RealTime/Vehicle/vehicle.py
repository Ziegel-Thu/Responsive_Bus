from Node.node import Node
from Routes.route import Route
from Parameters import vehicle_capacity

class Vehicle:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.route = Route()
        self.current_location = Node(0, 0, "")
                
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
        self.route.remove_node(node)
    
    def get_load(self):
        return self.route.get_load()
    
    def update_load(self, node: Node):
        self.route.update_load(node)
    