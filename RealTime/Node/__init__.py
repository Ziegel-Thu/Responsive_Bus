class Node:
    def __init__(self, demand_index=None, point_index=None):
        self.demand_index = demand_index  # 需求索引
        self.point_index = point_index    # 点位索引

    def __str__(self):
        return f"Node(demand_index={self.demand_index}, point_index={self.point_index})"

    def get_demand_index(self):
        return self.demand_index

    def get_point_index(self):
        return self.point_index




