from Node.node import Node

class Demand:
    def __init__(self, pickup_nodes: list[Node], delivery_nodes: list[Node], demand_load: int, start_time: float, end_time: float, index: int):
        self.pickup_nodes = pickup_nodes
        self.delivery_nodes = delivery_nodes
        self.demand_load = demand_load
        self.start_time = start_time
        self.end_time = end_time
        self.index = index

    def get_pickup_nodes(self):
        return self.pickup_nodes

    def get_delivery_nodes(self):
        return self.delivery_nodes

    def get_demand_load(self):
        return self.demand_load

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time
    
    def get_index(self):
        return self.index
    
class DemandList:
    def __init__(self, demandlist: list[Demand]):
        self.demandlist = demandlist
        
    def add_demand(self, demand: Demand):
        self.demandlist.append(demand)

    def get_demand(self, demand_index: int):
        return self.demandlist[demand_index]

    def get_all_demands(self):
        return self.demandlist

    def get_num_demands(self):
        return len(self.demandlist)

    def get_pickup_nodes_list(self):
        pickup_nodes_list : list[list[Node]] = []
        for demand in self.demandlist:
            pickup_nodes_list.append(demand.get_pickup_nodes())
        return pickup_nodes_list

    def get_delivery_nodes_list(self):
        delivery_nodes_list : list[list[Node]]= []
        for demand in self.demandlist:
            delivery_nodes_list.append(demand.get_delivery_nodes())
        return delivery_nodes_list

    def get_demand_load(self, demand_index: int):
        return self.demandlist[demand_index].get_demand_load()
    
    def get_demand_by_node(self, node: Node):
        node_demand_index = node.get_demand_index()
        return self.demandlist[node_demand_index]
