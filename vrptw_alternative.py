import random
from util import *
import os
import copy

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # le path du dossier 'VRPTW'

def start(instance_name):
    json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    data = load_instance(json_file=json_file)

    if data is None:
        print("Data not found")
        return
    return data

def generate_feasible_solution_nearest(data):
    solution = []
    vehicule = [0]
    capacity_current = 0
    time_current = 0

    clients_non_assigned = list(range(1, len(data["distance_matrix"][0]) ))  # 1 to 100

    while clients_non_assigned:
        client_current_id = vehicule[-1] if vehicule else 0  # Start from the depot for each vehicle
        selected_client_id = select_random_client(clients_non_assigned)
        selected_client = getClientName(selected_client_id)

        # Find the best position to insert the selected client in the route
        best_position = find_best_position(selected_client_id, vehicule, data)

        # Check if adding the client to the route violates constraints
        if (
            capacity_current + data[selected_client]["demand"] <= data["vehicle_capacity"]
            and time_current
            + data["distance_matrix"][client_current_id][selected_client_id]
            <= data[selected_client]["due_time"]
        ):

            # Insert the client into the route at the best position
            vehicule.insert(best_position, selected_client_id)
            capacity_current += data[selected_client]["demand"]
            time_current += (
                data["distance_matrix"][client_current_id][selected_client_id]
                + data[selected_client]["service_time"]
            )

            # Remove the client from the list of non-assigned clients
            clients_non_assigned.remove(selected_client_id)

        else:
            if vehicule[-1] != 0:
                vehicule.append(0)
                solution.append(vehicule[:])
            vehicule = [0]
            capacity_current = 0
            time_current = 0

    return solution
def find_best_position(selected_client_id, route, data):
    best_position = 1  # Start from the first position

    # Initialize variables for the best insertion cost and the current route length
    best_insertion_cost = float('inf')
    current_route_length = calculate_total_distance([route], data)

    # Iterate over possible positions in the route
    for position in range(1, len(route)):
        # Calculate the cost of inserting the client at the current position
        insertion_cost = calculate_insertion_cost(selected_client_id, position, route, data)

        # Check if the insertion is feasible and improves the route length
        if insertion_cost < best_insertion_cost:
            best_insertion_cost = insertion_cost
            best_position = position

    return best_position


def calculate_insertion_cost(selected_client_id, position, route, data):
    # Calculate the change in route length by inserting the selected client at the specified position
    prev_client_id = route[position - 1]
    next_client_id = route[position]
    selected_client = getClientName(selected_client_id)

    insertion_cost = (
        data["distance_matrix"][prev_client_id][selected_client_id]
        + data["distance_matrix"][selected_client_id][next_client_id]
        - data["distance_matrix"][prev_client_id][next_client_id]
    )

    # Check if the insertion satisfies time window constraints
    arrival_time = calculate_arrival_time(selected_client_id, prev_client_id, route, data)
    waiting_time = max(0, data[selected_client]["ready_time"] - arrival_time)
    time_penalty = waiting_time + data[selected_client]["service_time"]

    # Include the time penalty in the insertion cost
    insertion_cost += time_penalty

    return insertion_cost


def calculate_arrival_time(selected_client_id, prev_client_id, route, data):
    # Calculate the arrival time at the selected client in the route
    prev_client_index = route.index(prev_client_id)
    travel_time = data["distance_matrix"][route[prev_client_index]][selected_client_id]
    arrival_time = route_time_at_index(prev_client_index, route, data) + travel_time
    return arrival_time


def route_time_at_index(index, route, data):
    # Calculate the total time spent on the route up to the specified index
    time = 0
    for i in range(index):
        current_client_id = route[i]
        next_client_id = route[i + 1]
        time += data["distance_matrix"][current_client_id][next_client_id] + data[getClientName(current_client_id)]["service_time"]
    return time

def initialize_population_nearest(taille_population, data):
    population = []

    while len(population) < taille_population:
        solution = generate_feasible_solution_nearest(data)

        # Check if the solution is unique in the population
        if solution not in population:
            population.append(solution)

    return population

def initialize_population(taille_population, data):
    population = []

    while len(population) < taille_population:
        solution = generate_feasible_solution(data)

        # Check if the solution is unique in the population
        if solution not in population:
            population.append(solution)

    return population

data = start('C103')

population_size = 5
initial_population = initialize_population(population_size, data)

print("Initial Population:\n")
for i, solution in enumerate(initial_population):
    print(f"Solution {i + 1}: {solution}")
    print("\n")

print("\n------------------------------------")
print("\nObjectives Scores:")
for i, solution in enumerate(initial_population):
    score, total_distance, total_waiting_time, nombre_de_véhicules = objective(solution, data, vehicules_number_weigth=0.5)
    print(f"Solution {i + 1}: Score: {score}, Total Distance: {total_distance}, Total Waiting Time: {total_waiting_time}, Number of Vehicles: {nombre_de_véhicules}")
