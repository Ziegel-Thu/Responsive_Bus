import numpy as np

class Calculator:
    def __init__(self, distance_matrix, time_windows, penalty_coefficient):
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.penalty_coefficient = penalty_coefficient

    def total_distance(self, routes):
        total_distance = 0
        final_depot = 2 * len(routes[0].get_route()) + 1
        for route in routes:
            route_nodes = route.get_route()
            if len(route_nodes) > 0:
                total_distance += self.distance_matrix[0][route_nodes[0]]
                for i in range(1, len(route_nodes)):
                    total_distance += self.distance_matrix[route_nodes[i - 1]][route_nodes[i]]
                total_distance += self.distance_matrix[route_nodes[-1]][final_depot]
        return total_distance

    def total_penalty(self, routes):
        total_penalty = 0
        final_depot = 2 * len(routes[0].get_route()) + 1
        for route in routes:
            route_nodes = route.get_route()
            current_time = 0
            for i in range(len(route_nodes)):
                if i == 0:
                    current_time += self.distance_matrix[0][route_nodes[i]]
                else:
                    current_time += self.distance_matrix[route_nodes[i - 1]][route_nodes[i]]
                
                if route_nodes[i] == 0 or route_nodes[i] == final_depot:
                    continue

                if current_time > self.time_windows[route_nodes[i]][1]:
                    total_penalty += (current_time - self.time_windows[route_nodes[i]][1]) * self.penalty_coefficient
                if current_time < self.time_windows[route_nodes[i]][0]:
                    current_time = self.time_windows[route_nodes[i]][0]
        return total_penalty

    def calculate_route_cost(self, route):
        total_distance = 0
        total_penalty = 0
        current_time = 0
        route_nodes = route.get_route()
        final_depot = 2 * len(route_nodes) + 1
        
        for i in range(len(route_nodes)):
            if i == 0:
                total_distance += self.distance_matrix[0][route_nodes[i]]
                current_time += self.distance_matrix[0][route_nodes[i]]
            else:
                total_distance += self.distance_matrix[route_nodes[i - 1]][route_nodes[i]]
                current_time += self.distance_matrix[route_nodes[i - 1]][route_nodes[i]]
                
            if route_nodes[i] == 0 or route_nodes[i] == final_depot:
                continue

            if current_time > self.time_windows[route_nodes[i]][1]:
                total_penalty += (current_time - self.time_windows[route_nodes[i]][1]) * self.penalty_coefficient
            if current_time < self.time_windows[route_nodes[i]][0]:
                current_time = self.time_windows[route_nodes[i]][0]

        total_distance += self.distance_matrix[route_nodes[-1]][final_depot]
        return total_distance, total_penalty

    def calculate_total_cost(self, routes):
        return self.total_distance(routes) + self.total_penalty(routes)
