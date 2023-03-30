import pandas as pd
import random
from itertools import permutations
from functools import reduce
from flask import Flask, request, jsonify

app = Flask(__name__)


def read_xlsx(file_path):
    data = pd.read_excel(file_path, engine='openpyxl',
                         header=None, index_col=0)
    data = data.transpose()
    return data


def preprocess_data(data):
    data.columns = ['Name', 'BA_Rank', 'Last_5_Rank', 'Number',
                    'Average', 'Last_5', 'Hits', 'At_Bats'] + list(data.columns[8:])
    data.drop(['Number'], axis=1, inplace=True)
    data.dropna(subset=['Name'], inplace=True)  # Drop rows with missing names
    return data


def fitness_function(order, data):
    lineup = data.set_index('Name').loc[list(order)]
    return lineup['Average'].sum()


def create_initial_population(players, population_size, lineup_size):
    population = []
    for _ in range(population_size):
        individual = random.sample(players, lineup_size)
        population.append(individual)
    return population


def crossover(parent1, parent2):
    size = len(parent1)
    crossover_point = random.randint(1, size - 1)
    child = parent1[:crossover_point] + \
        [x for x in parent2 if x not in parent1[:crossover_point]]
    return child


def mutate(individual):
    index1, index2 = random.sample(range(len(individual)), 2)
    individual[index1], individual[index2] = individual[index2], individual[index1]


def genetic_algorithm(data, lineup_size, population_size=100, generations=1000, mutation_rate=0.1):
    players = data['Name'].tolist()
    population = create_initial_population(
        players, population_size, lineup_size)

    for _ in range(generations):
        population = sorted(population, key=lambda order: fitness_function(
            order, data), reverse=True)
        new_population = population[:2]

        for _ in range(population_size // 2 - 1):
            parent1, parent2 = random.sample(population[:10], 2)
            child = crossover(parent1, parent2)

            if random.random() < mutation_rate:
                mutate(child)

            new_population.append(child)

        population = new_population

    best_order = population[0]
    return best_order


@app.route('/api/optimal_batting_order', methods=['GET'])
def generate_optimal_batting_order():
    lineup_size = int(request.args.get('lineup_size', 10))
    data = read_xlsx("batting-lineup.xlsx")
    processed_data = preprocess_data(data)
    optimal_order = genetic_algorithm(processed_data, lineup_size)
    return jsonify({'optimal_batting_order': optimal_order})


if __name__ == '__main__':
    app.run(debug=True)
