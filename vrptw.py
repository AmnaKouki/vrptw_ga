import os
import io
import random
from util import *
from csv import DictWriter
from remplacement import *

from util import make_directory_for_file, exist, load_instance, merge_rules

BASE_DIR =os.path.abspath(os.path.dirname(__file__)) ## le path du dossier 'VRPTW'

def start(instance_name):
 
    json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    data = load_instance(json_file=json_file)

    if data is None:
        print("Data not found")
        return
    return data
    



#----------------------------- Début Traitement ------------------------------#


data = start('C103')

taille_population = 10
initial_population = initialize_population(taille_population, data)

print("Initial Population:\n")
for i, solution in enumerate(initial_population):

    print(f"Solution {i + 1}: {solution}")
    print("\n")
    
    
print("\n------------------------------------")
print("\nObjectives Scores:")
for i, solution in enumerate(initial_population):  # Assuming 'solutions' is the list of solutions you want to evaluate
    score, total_distance, total_waiting_time, nombre_de_véhicules = objective(solution, data, vehicules_number_weigth=0.5)
    print(f"Solution {i + 1}: Score: {score}, Total Distance: {total_distance}, Total Waiting Time: {total_waiting_time}, Number of Vehicles: {nombre_de_véhicules}")
    
#Selection 
fitness_scores = [calculate_fitness(solution, data) for solution in initial_population]
selected_solution, solution_number = roulette_selection(initial_population, fitness_scores)
print("\n------------------------------------")
print(f"Selected Solution {solution_number +1}: {selected_solution }")
print("\n------------------------------------")






# #parent1 = random.choice(initial_population)
parent1 = selected_solution
parent2, parent2_number = roulette_selection(initial_population, fitness_scores)

print("\n------------------------------------")
print("   CrossOver :")
print("------------------------------------")
print(f"Parent 1: {parent1} \n")
print(f"Parent 2: {parent2}\n")

# Apply crossover
child1, child2 = order_crossover1(parent1, parent2,data)
print(f"Child 1 after crossover: {child1}\n")
print(f"Child 2 after crossover: {child2}\n")


#Apply crossover with  rate =0.8
# if will_crossover():
#     print("\n------------------------------------")
#     print("\n-------------SUCCESSFULL CX PARIALLY MATCHED CROSSOVER------------------")

#     child1,child2 = cx_partially_matched(parent1, parent2,data)
#     print(f"Child 1 after pmx crossover  : {child1}\n")
#     print(f"Child 2 after pmx crossover  : {child2}\n")
# else :
#     print("\n------------------------------------")
#     print("\n------------- FAILED CX PARTIALLY MATCHED CROSSOVER ------------------")

#     child1=parent1
#     child2=parent2


# # Apply crossover with  rate =0.8
# if will_crossover():
#     print("\n------------------------------------")
#     print("\n-------------SUCCESSFULL ORDER CROSSOVER------------------")

#     ch1,ch2 = order_crossover(parent1, parent2,data)
#     print(f"Child 1 after crossover order : {ch1}\n")
#     print(f"Child 2 after crossover order : {ch2}\n")
# else :
#     print("\n------------------------------------")
#     print("\n------------- FAILED ORDER CROSSOVER ------------------")

#     ch1=parent1
#     ch2=parent2

# print("same child",ch1==ch2)

# # TODO : cross over alternative
#child = order_crossover(parent1, parent1)
# print("\n------------------------------------")
# print(f"Child 1 after crossover order : \n")
# for i, solution in enumerate(child):

#     print(f"Solution {i + 1}: {solution}")
#     print("\n")


#is_child_feasible = is_solution_valid(child1, data)
#print(f"Child 1 is feasible: {is_child_feasible}")

# print("\n------------------------------------")
# print("   Mutation :")
# print("------------------------------------")

mutation_rate = 0.3

mutated_solution = mutation_with_rate(list(child1), data, mutation_rate)
mutated_solution2 = mutation_with_rate(list(child2), data, mutation_rate)
print("------------------------------------")
print("Child 1 mutée :", mutated_solution)




# #TODO  - mutation result -
# #Ensure feasibility of children
# is_child1_feasible = is_solution_valid(mutated_solution, data)
# is_child2_feasible = is_solution_valid(mutated_solution2, data)
# print("**********************************************")
# print(f"Child 1 is feasible: {is_child1_feasible}")
# print(f"Child 2 is feasible: {is_child2_feasible}")



nouvelle_population = remplacement_avec_elitisme(initial_population, [mutated_solution, child1], 0.4,data,distance_weight=0.7, waiting_time_weight=0.3)

print("\n------------------------------------")
print("   Remplacement avec elitisme :")
print("------------------------------------")
for i, solution in enumerate(nouvelle_population):
    print(f"Solution {i + 1}: {solution}")
    print("\n")
