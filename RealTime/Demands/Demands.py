class Demands:
    def __init__(self):
        self.demand_pairs = []  # [(pickup_points, delivery_points), ...]
        self.demands_values = []  # 每对需求的载重值

    def add_demand(self, pickup_points, delivery_points, demand_value):
        self.demand_pairs.append((pickup_points, delivery_points))
        self.demands_values.append(demand_value)

    def get_pickup_points(self, demand_index):
        return self.demand_pairs[demand_index][0]

    def get_delivery_points(self, demand_index):
        return self.demand_pairs[demand_index][1]

    def get_demand_value(self, demand_index):
        return self.demands_values[demand_index]

    def get_all_demands(self):
        return self.demand_pairs

    def get_num_demands(self):
        return len(self.demand_pairs)

    def remove_demand(self, demand_index):
        self.demand_pairs.pop(demand_index)
        self.demands_values.pop(demand_index)
