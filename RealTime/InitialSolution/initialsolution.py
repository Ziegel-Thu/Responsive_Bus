import numpy as np
from Routes.Route import Route
from Demands.demands import Demands

class InitialSolution:
    def __init__(self, num_vehicles, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient, demands):
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.penalty_coefficient = penalty_coefficient
        self.demands = demands
        self.routes = self.greedy_initial_solution()
        print("InitialSolution Done")
        print("InitialSolution:", [route.get_route() for route in self.routes])

    def greedy_initial_solution(self):
        routes = [Route() for _ in range(self.num_vehicles)]
        unassigned_demands = list(range(self.demands.get_num_demands()))
        vehicle_load = [0] * self.num_vehicles
        current_locations = [0] * self.num_vehicles

        while unassigned_demands:
            for vehicle in range(self.num_vehicles):
                if not unassigned_demands:
                    continue

                best_demand = None
                best_pickup = None
                best_delivery = None
                best_cost = float('inf')

                for demand_idx in unassigned_demands:
                    pickup_points = self.demands.get_pickup_points(demand_idx)
                    delivery_points = self.demands.get_delivery_points(demand_idx)
                    demand_value = self.demands.get_demand_value(demand_idx)

                    if vehicle_load[vehicle] + demand_value > self.vehicle_capacity:
                        continue

                    # 找最优的接送点组合
                    for pickup in pickup_points:
                        for delivery in delivery_points:
                            pickup_cost = self.distance_matrix[current_locations[vehicle]][pickup]
                            delivery_cost = self.distance_matrix[pickup][delivery]
                            total_cost = pickup_cost + delivery_cost

                            if total_cost < best_cost:
                                best_cost = total_cost
                                best_demand = demand_idx
                                best_pickup = pickup
                                best_delivery = delivery

                if best_demand is not None:
                    # 添加接客点
                    routes[vehicle].add_node(best_pickup)
                    routes[vehicle].update_load(self.demands.get_demand_value(best_demand))
                    vehicle_load[vehicle] += self.demands.get_demand_value(best_demand)
                    current_locations[vehicle] = best_pickup

                    # 添加送客点
                    routes[vehicle].add_node(best_delivery)
                    routes[vehicle].update_load(-self.demands.get_demand_value(best_demand))
                    vehicle_load[vehicle] -= self.demands.get_demand_value(best_demand)
                    current_locations[vehicle] = best_delivery

                    unassigned_demands.remove(best_demand)

        # 添加终点站
        final_depot = len(self.distance_matrix) - 1
        for route in routes:
            if not route.is_empty() and route.get_last_node() != final_depot:
                route.add_node(final_depot)

        print("The initial route is:", [route.get_route() for route in routes])
        return routes

# 这里可以添加其他辅助函数或类
