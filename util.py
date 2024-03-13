
import io
from json import dump, load
import math
from operator import attrgetter
import os
import random
#from util import make_directory_for_file, exist, load_instance, merge_rules

BASE_DIR =os.path.abspath(os.path.dirname(__file__)) ## le path du dossier 'VRPTW'


def start(instance_name):
 
    json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    data = load_instance(json_file=json_file)

    if data is None:
        print("Data not found")
        return
    return data
    
def calculate_distance1(customer1, customer2):
    aux = (customer1['coordinates']['x'] - customer2['coordinates']['x'])**2 + \
        (customer1['coordinates']['y'] - customer2['coordinates']['y'])**2
    return math.sqrt(aux)

def calculate_distance(customer1, customer2,data):
    aux = (data[customer1]['coordinates']['x'] - data[customer2]['coordinates']['x'])**2 + \
        (data[customer1]['coordinates']['y'] - data[customer2]['coordinates']['y'])**2
    return math.sqrt(aux)
    
            
def make_directory_for_file(path_name):
    try:
        os.makedirs(os.path.dirname(path_name))
    except OSError:
        pass

def load_instance(json_file):
    if exist(path=json_file):
        with io.open(json_file, 'rt', encoding='utf-8', newline='') as file_object:
            return load(file_object)
    return None

def exist(path):
    if os.path.exists(path):
        return True
    return False

def merge_rules(rules):
    is_fully_merged = True
    for round1 in rules:
        if round1[0] == round1[1]:
            rules.remove(round1)
            is_fully_merged = False
        else:
            for round2 in rules:
                if round2[0] == round1[1]:
                    rules.append((round1[0], round2[1]))
                    rules.remove(round1)
                    rules.remove(round2)
                    is_fully_merged = False
    return rules, is_fully_merged

def getClientName(client_id):
    return f'customer_{client_id}'
def getClientId(client_name):
    return int(client_name.split('_')[1])


def select_random_client(clients_non_assigned):
    return random.choice(clients_non_assigned)

def getVehicleNumber(solution):
    return len(solution)

def calculate_total_distance(solution, data):
    total_distance = 0
    for route in solution:
        for i in range(len(route) - 1):
            total_distance += calculate_distance(getClientName(route[i]) , getClientName(route[i + 1]), data)
    return total_distance


def calculate_waiting_time_per_route(route, data):
    total_waiting_time = 0
    current_time = 0

    for client in route[1:]:
        current_client_id = client
        current_client = getClientName(current_client_id)
        
        #Calculate the arrival time at the current client. It is the sum of the current time and the time taken to travel from the previous client to the current one 
        arrival_time = current_time + data["distance_matrix"][current_client_id -1][current_client_id ]
        #waiting_time: Calculate the waiting time at the current client. It is the maximum of 0 and the difference between the start of the time window for the client (data["time_windows"][client][0]) and the calculated arrival time
        waiting_time = max(0, data[current_client]['ready_time'] - arrival_time) #exemple arrival fi 6 w houwa start=7 max bin 0 et 7-6=1 """" [0 ]start
        #If the vehicle arrives within the time window or after it starts, the waiting time is 0. Otherwise, it's the difference between the start of the time window and the arrival time.
        total_waiting_time += waiting_time
        current_time = max(arrival_time, data[current_client]['ready_time']) + data[current_client]['service_time']
#Update the current time. It is the maximum of the arrival time and the start of the time window for the client, plus the service time required at the current client.
#Met à jour le temps actuel. C'est le maximum de l'heure d'arrivée et du début de la plage horaire du client, plus le temps de service requis au client actuel.    
    
    return total_waiting_time


def calculate_waiting_time(solution,data):
    total_waiting_time = 0
    for route in solution:
        total_waiting_time += calculate_waiting_time_per_route(route, data)
    return total_waiting_time


# Fonction pour calculer le score de fitness d'une solution (distance totale parcourue)
def calculate_fitness(solution, data,distance_weight=0.7, waiting_time_weight=0.3):
    total_distance = 0
    total_waiting_time = 0

    total_distance += calculate_total_distance(solution, data)
    total_waiting_time += calculate_waiting_time(solution, data)

    # Combine objectives with weights
    score = distance_weight * total_distance + waiting_time_weight * total_waiting_time

    return score


# Fonction objectif qui minimise la distance totale parcourue et le nombre de véhicules utilisés
def objective(solution, data,vehicules_number_weigth=0.7):
    total_distance=0
    total_waiting_time=0
    nombre_de_véhicules = len(solution)  # Nombre de routes dans la solution

    total_waiting_time += calculate_waiting_time(solution, data)
    total_distance += calculate_total_distance(solution, data)


    score=calculate_fitness(solution, data,distance_weight=0.3, waiting_time_weight=0.2)
    score=score+nombre_de_véhicules*vehicules_number_weigth

    return score,total_distance,total_waiting_time,nombre_de_véhicules

def generate_feasible_solution(data):
    solution = [] 
    vehicule = [0] 
    capacity_current = 0
    time_current = 0

    clients_non_assigned = list(range(1, len(data["distance_matrix"][0] )-1)) # 1 to 100
    while clients_non_assigned:
        client_current_id = vehicule[-1] if vehicule else 0  # Start from the depot for each vehicle
        client_candidat_id = select_random_client(clients_non_assigned)
        
        client_current = getClientName(client_current_id)
        client_candidat = getClientName(client_candidat_id)
        
        
        # Check if adding the client to the route violates constraints
        if (capacity_current + data[client_candidat]['demand'] <= data['vehicle_capacity'] and
                time_current +   data['distance_matrix'][client_current_id][client_candidat_id] <= data[client_candidat]['due_time']):

            # Add the client to the solution for the current vehicle
            vehicule.append(client_candidat_id)
            capacity_current += data[client_candidat]['demand']
            time_current +=    data['distance_matrix'][client_current_id][client_candidat_id] + data[client_candidat]['service_time']

            # Remove the client from the list of non-assigned clients
            clients_non_assigned.remove(client_candidat_id)
            

        else:
            #look for the closest client
            closest_client_id = min(clients_non_assigned, key=lambda client: calculate_distance(client_current, getClientName(client), data))
    
            closest_client = getClientName(closest_client_id)
     
            if (capacity_current +  data[closest_client]['demand'] <= data["vehicle_capacity"] and
     
                    time_current + data['distance_matrix'][client_current_id][closest_client_id] <= data[closest_client]['due_time'] ):
                vehicule.append(closest_client_id)
                
                #data['distance_matrix'][client_current_id][closest_client_id] + data[closest_client]['service_time']

                
                capacity_current += data[closest_client]['demand']
                time_current += data['distance_matrix'][client_current_id][closest_client_id] + data[closest_client]['service_time']
                clients_non_assigned.remove(closest_client_id)
            else:
            
                if vehicule[-1] != 0: 
                    vehicule.append(0) 
                    solution.append((vehicule[:]))
                vehicule = [0]
                capacity_current = 0
                time_current = 0
            
    # print("----------------------------")
    # print(solution)
    return solution

def initialize_population(taille_population, data):
    population = []

    while len(population) < taille_population:
        solution = generate_feasible_solution(data)
          # Check if the solution is unique in the population
        if solution not in population:
            population.append(solution)

    return population


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

def initialize_population_insertion(taille_population, data):
    population = []

    while len(population) < taille_population:
        solution = generate_feasible_solution_nearest(data)

        # Check if the solution is unique in the population
        if solution not in population:
            population.append(solution)

    return population


def roulette_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selected_index = None

    # Generate a random number between 0 and the total fitness
    random_value = random.uniform(0, total_fitness)
    
    # Iterate through the population and select the individual whose cumulative fitness exceeds the random value
    cumulative_fitness = 0
    for i, score in enumerate(fitness_scores):
        cumulative_fitness += score
        if cumulative_fitness >= random_value:
            selected_index = i
            break

    # Return the selected individual from the population
    selected_individual = population[selected_index]
    return selected_individual ,selected_index


# mutation 
def mutation_inversion(solution, data):

    # Copie de la solution pour éviter de modifier l'original
    mutated_solution = solution.copy()  
    
    # Choix aléatoire d'une route dans la solution
    random_route_index = random.randint(0, len(mutated_solution) - 1)
    random_route = mutated_solution[random_route_index]
    
    # Choix de deux indices distincts aléatoires dans la route
    idx1, idx2 = random.sample(range(1, len(random_route)), 2)
    
    # Tri des indices en ordre croissant
    idx1, idx2 = sorted([idx1, idx2])
    
    # Vérification des cas particuliers
    if idx1 == idx2:
        # Si les indices sont les mêmes, choisir un autre indice pour idx2
        idx2 = random.choice(range(idx1 + 1, len(random_route)))
    
    # Inversion de l'ordre des clients entre idx1 et idx2
    random_route[idx1:idx2+1] = reversed(random_route[idx1:idx2+1])
    
    # Vérification de la validité de la solution mutée
    if not is_solution_valid(list(mutated_solution), data):
        # Si la solution mutée est invalide, réinitialiser la solution à sa valeur d'origine
        mutated_solution = solution.copy()
    
    return mutated_solution


# vérifier la validité de la solution mutée par inversion
def is_solution_valid(solution, data):
   # solution = list(solution)[0]
    solution = list(solution)
    for route in solution:
        capacity = 0
        time_current = 0
        
        for i in range(len(route) - 1):
            client_current_id = route[i]
            client_candidat_id = route[i + 1]

            client_current = getClientName(client_current_id)
            client_candidat = getClientName(client_candidat_id)

            
            demand = data[client_candidat]['demand']
            service_time = data[client_candidat]['service_time']
            
            ready_time = data[client_candidat]['ready_time']
            due_time = data[client_candidat]['due_time']
            
            # Check if adding the client to the route violates constraints
            if (capacity + demand <= data['vehicle_capacity'] and
                    time_current + data['distance_matrix'][client_current_id][client_candidat_id] <= due_time):

                # Update the capacity and time for the current vehicle
                capacity += demand
                time_current += data['distance_matrix'][client_current_id][client_candidat_id] + service_time

            else:
                # The solution is invalid if constraints are violated
                return False

        # Check if the last delivery is the depot
        if route[-1] != 0:
            return False

    return True

def mutation_with_rate(solution, data, mutation_rate):
    mutated_solution = mutation_with_rate1(solution, data, mutation_rate)
    if is_solution_valid(mutated_solution, data):
        return mutated_solution
    else:
        return solution
   
    
    
    return mutated_solution
def mutation_with_rate1(solution, data, mutation_rate):
    mutated_solution = solution.copy()
    
    # Vérification du taux de mutation
    if random.random() < mutation_rate:
        # Appliquer la mutation inversion avec la fonction mutation_inversion
        mutated_solution = mutation_inversion(mutated_solution, data)
    
    return mutated_solution
    

def cx_partially_matched(ind1, ind2, data):
    child1, child2 = cx_partially_matched1(ind1, ind2)
    if not is_solution_valid(child1, data):
        child1 = ind1
    if not is_solution_valid(child2, data) :
        child2= ind2
   
    return child1, child2
    
  
def cx_partially_matched1(ind1, ind2):
      # Select two random crossover points
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    
    # Extract the parts to be swapped
    part1 = ind2[cxpoint1:cxpoint2+1]
    part2 = ind1[cxpoint1:cxpoint2+1]
    # Create a mapping between the elements in the two parts
    mapping = list(zip(part1, part2))

    # Merge the mapping until it's fully merged
    is_fully_merged = False
    while not is_fully_merged:
        mapping, is_fully_merged = merge_rules(rules=mapping)
      # Create the reverse mapping
    reverseMapping = {tuple(rule[1]): tuple(rule[0]) for rule in mapping}

    # Apply the mapping to produce offspring
    ind1 = [gene if gene not in part2 else reverseMapping[gene] for gene in ind1[:cxpoint1]] + part2 + \
           [gene if gene not in part2 else reverseMapping[gene] for gene in ind1[cxpoint2+1:]]
    
    ind2 = [gene if gene not in part1 else mapping[gene] for gene in ind2[:cxpoint1]] + part1 + \
           [gene if gene not in part1 else mapping[gene] for gene in ind2[cxpoint2+1:]]
    
    return ind1, ind2

def remplacement_avec_elitisme(population, enfants, taux_elitisme, data, distance_weight=0.7, waiting_time_weight=0.3):
    # Taille de la population
    taille_population = len(population)
    
    # Nombre d'élites à conserver
    nombre_elites = round(taux_elitisme * taille_population)
    
    # Fonction de fitness interne utilisant calculate_fitness avec les paramètres donnés
    def fitness_function(solution):
        return calculate_fitness(solution, data, distance_weight, waiting_time_weight)
    
    # Calcul des scores de fitness pour la population actuelle et les enfants
    population_fitness = [fitness_function(solution) for solution in population]
    enfants_fitness = [fitness_function(enfant) for enfant in enfants]
    
    # Combinaison des individus de la population actuelle et des enfants
    individus_combines = population + enfants
    individus_fitness_combines = population_fitness + enfants_fitness
    
    # Trie des individus combinés en fonction de leur score de fitness
    individus_tries = [individu for _, individu in sorted(zip(individus_fitness_combines, individus_combines), reverse=True)]
    
    # Sélection des élites
    elites = individus_tries[:nombre_elites]
    
    # Formation de la nouvelle population en sélectionnant les meilleurs individus (à l'exclusion des élites)
    nouvelle_population = individus_tries[nombre_elites:]
    
    # Ajout des élites de la population initiale à la nouvelle population
    for elite_solution in population[:nombre_elites]:
        nouvelle_population.append(elite_solution)
    
    return nouvelle_population




def order_crossover(ind1, ind2, data):
    child1, child2 = order_crossover1(ind1, ind2)
    while not (is_solution_valid(child1, data) ) and (not is_solution_valid(child2,data) ) :
         child1, child2 = order_crossover1(ind1, ind2)
    return child1, child2
def order_crossover1(parent1, parent2):
    # Choix de deux points de crossover
    crossover_points = sorted(random.sample(range(min(len(parent1), len(parent2))), 2))

    # Copie des sous-séquences entre les points de crossover
    subsequence_parent1 = parent1[crossover_points[0]:crossover_points[1] + 1]
    subsequence_parent2 = parent2[crossover_points[0]:crossover_points[1] + 1]

    # Initialisation des enfants avec des listes contenant des marqueurs
    child1 = [None] * len(parent1)
    child2 = [None] * len(parent2)

    # Copie des sous-séquences dans les enfants
    child1[crossover_points[0]:crossover_points[1] + 1] = subsequence_parent1
    child2[crossover_points[0]:crossover_points[1] + 1] = subsequence_parent2

    # Remplissage des gaps dans les enfants
    fill_gaps(child1, parent2, crossover_points)
    fill_gaps(child2, parent1, crossover_points)

    return list(child1), list(child2)



def fill_gaps(child, parent, crossover_points):
    # Remplissage des gaps avec les éléments du parent non inclus dans la sous-séquence
    position = crossover_points[1] + 1

    for element in parent:
        if element not in child:
            child[position % len(parent)] = element
            position += 1


# A higher value increases the likelihood of crossover, while a lower value decreases it. 
def will_crossover(taux_croisement=0.8):
    return random.random() <taux_croisement


