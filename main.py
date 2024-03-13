import time
from util import *
# def algo_genetique(data, taille_population,nbr_iteration, distance_weight, waiting_time_weight, vehicules_number_weigth, taux_croisement, mutation_rate, taux_elitisme):
#     best_solutions = []
#     population = initialize_population(taille_population,data)
#     #population = initialize_population_nearest(taille_population,data)
#     affichage_resultat(population, vehicules_number_weigth)
    
    
#     for i in range(nbr_iteration):
#         fitness_scores = [calculate_fitness(solution, data) for solution in population]
        
#         parent1 ,parent1_number= roulette_selection(population, fitness_scores)
#         parent2 , parent2_number = roulette_selection(population, fitness_scores)
        
#         if will_crossover(taux_croisement):
#             ch1,ch2 = cx_partially_matched(parent1, parent2,data)
#             #ch1,ch2 = order_crossover(parent1, parent2,data)
#         else :
#             ch1=parent1
#             ch2=parent2
#         mutated_solution = mutation_with_rate(list(ch1), data, mutation_rate)
#         mutated_solution2 = mutation_with_rate(list(ch2), data, mutation_rate)
#         nouvelle_population = remplacement_avec_elitisme(population, [mutated_solution, mutated_solution2], taux_elitisme, data, distance_weight, waiting_time_weight)
#         population = nouvelle_population
        
#         sol =affichage_resultat(population, vehicules_number_weigth)
#         best_solutions.append(sol)
        
#     #affichage_resultat(population, vehicules_number_weigth)
#     # print("Best Solution: ")
#     # print(best_solutions)
#     best_solution(best_solutions)
#     return population

def algo_genetique_with_time_limit(data, taille_population, distance_weight, waiting_time_weight, vehicules_number_weigth, taux_croisement, mutation_rate, taux_elitisme, time_limit):
    
    population = initialize_population(taille_population, data)
    #population = initialize_population_insertion(taille_population,data)
    start_time = time.time()
    best_solutions = []
    #print(affichage_resultat(population, vehicules_number_weigth))
    
    iteration = 0
    while True:
        iteration += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            break  # Exit the loop if time limit is reached
        
        fitness_scores = [calculate_fitness(solution, data) for solution in population]
        
        parent1, parent1_number = roulette_selection(population, fitness_scores)
        parent2, parent2_number = roulette_selection(population, fitness_scores)
        
        if will_crossover(taux_croisement):
            ch1, ch2 = cx_partially_matched(parent1, parent2, data)
            #ch1,ch2 = order_crossover(parent1, parent2,data)
        else:
            ch1 = parent1
            ch2 = parent2
        
        mutated_solution = mutation_with_rate(list(ch1), data, mutation_rate)
        mutated_solution2 = mutation_with_rate(list(ch2), data, mutation_rate)
        nouvelle_population = remplacement_avec_elitisme(population, [mutated_solution, mutated_solution2], taux_elitisme, data, distance_weight, waiting_time_weight)
        population = nouvelle_population
        
        sol = affichage_resultat(population, vehicules_number_weigth)
        best_solutions.append(sol)
    
    best_solution(best_solutions)
    return population
    
def best_solution(solutions):


    nb_vehicule = solutions[0]['Number_of_Vehicles']
    min_total_distance = solutions[0]['Total_Distance']
    min_total_waiting_time = solutions[0]['Total_Waiting_Time']
    min_score = solutions[0]['Score']
    
    
    for i,solution in enumerate(solutions): 
        if((solution["Number_of_Vehicles"] < nb_vehicule) and (solution["Score"] < min_score) ):
            
            nb_vehicule = solution["Number_of_Vehicles"]
            min_score = solution["Score"]
            min_total_distance = solution["Total_Distance"]
            min_total_waiting_time = solution["Total_Waiting_Time"]
    
    print("Best Solution: ", solution['solution'] )        
    print(f"Solution: Score: {min_score}, Total Distance: {min_total_distance}, Total Waiting Time: {min_total_waiting_time}, Number of Vehicles: {nb_vehicule}")

def affichage_resultat(population, vehicules_number_weigth):
    nb_vehicule = 500
    min_total_distance=999999
    min_total_waiting_time = 999999
    min_score = 999999
    best_sol = []
    for i, solution in enumerate(population):  
        score, total_distance, total_waiting_time, nombre_de_véhicules = objective(solution, data, vehicules_number_weigth)
        if(nombre_de_véhicules < nb_vehicule and score < min_score ):
            nb_vehicule = nombre_de_véhicules
            min_score = score
            min_total_distance = total_distance
            min_total_waiting_time = total_waiting_time
            best_sol = solution
        #print(f"Solution {i + 1}: Score: {score}, Total Distance: {total_distance}, Total Waiting Time: {total_waiting_time}, Number of Vehicles: {nombre_de_véhicules}")
       
    # print("Best Solution: ")
    # print(f"Score: {min_score}, Total Distance: {min_total_distance}, Total Waiting Time: {min_total_waiting_time}, Number of Vehicles: {nb_vehicule}")
    bestSol= {
        'Score': min_score,
        'Total_Distance': min_total_distance,
        'Total_Waiting_Time': min_total_waiting_time,
        'Number_of_Vehicles': nb_vehicule,
        'solution': best_sol
        
    }
    return bestSol
         
 

data = start('RC202')
taille_population = 100
time_limit = 3
nbr_iteration = 50

distance_weight = 0.7 # selection
waiting_time_weight = 0.6 # selection
vehicules_number_weight=0.9 # fonction objective
mutation_rate = 0.2 #0.05
taux_croisement=0.5 #0.8
taux_elitisme = 0.1


algo_genetique_with_time_limit(data, taille_population, distance_weight, waiting_time_weight, vehicules_number_weight,
                               taux_croisement, mutation_rate, taux_elitisme, time_limit)       
# algo_genetique(data, taille_population, nbr_iteration, distance_weight, waiting_time_weight, vehicules_number_weigth, taux_croisement, mutation_rate, taux_elitisme)

