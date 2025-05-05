import pandas as pd

Penalty_Coefficient = 3
num_vehicles = 5
vehicle_capacity = 100
distance_matrix = pd.read_csv('Data/wangjing-newdis-upflated.csv').values
initial_temperature = 1000
cooling_rate = 0.99
max_iterations = 500
max_recent_costs = 10
identical_solution_threshold = 3
maximum_time_limit = 6


