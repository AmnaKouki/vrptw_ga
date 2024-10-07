# VRPTW Solver using Genetic Algorithm

This project is a Python-based implementation of a Genetic Algorithm (GA) to solve the Vehicle Routing Problem with Time Windows (VRPTW). The VRPTW is a variant of the Vehicle Routing Problem (VRP), where each customer has a specific time window in which they must be visited.


The goal is to minimize the total travel distance while ensuring all customers are visited within their specified time windows.

## Problem Overview

The **Vehicle Routing Problem with Time Windows (VRPTW)** involves finding optimal routes for a fleet of vehicles to serve a set of customers, each with specific time windows. The problem becomes complex due to the need to satisfy both routing efficiency and time constraints.




## Genetic Algorithm Workflow

1. **Generate Initial Population**: Randomly generate potential solutions (routes).
2. **Evaluation**: Calculate the fitness of each chromosome based on total distance and time window satisfaction.
3. **Selection**: Choose the top-performing chromosomes for crossover based on fitness.
4. **Crossover**: Perform crossover between selected chromosomes to generate new offspring.
5. **Mutation**: Introduce small random changes in some chromosomes to maintain diversity.
6. **Evaluation and Replacement**: Re-evaluate the population and replace less fit individuals with better solutions.
7. **Choose Best Chromosome**: After a defined number of generations, the best solution (chromosome) is selected as the final result.


