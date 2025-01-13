import pandas as pd

Penalty_Coefficient = 3.14
num_vehicles = 3
vehicle_capacity = 100
distance_matrix = pd.read_csv('Data/distance_matrix.csv').values
initial_temperature = 1000
cooling_rate = 0.99
max_iterations = 10000

