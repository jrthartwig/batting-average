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
                    'Average', 'Last_5', 'Hits', 'At_Bats', 'Doubles', 'Doubles %'] + list(data.columns[10:])
    data.drop(['Number'], axis=1, inplace=True)
    data.dropna(subset=['Name'], inplace=True)  # Drop rows with missing names
    return data


def fitness_function(order, data):
    lineup = data.set_index('Name').loc[list(order)]
    outs_remaining = 3
    runs = 0

    for player in order:
        player_stats = lineup.loc[player]
        at_bats = int(player_stats['At_Bats'])
        hits = int(player_stats['Hits'])
        doubles = int(player_stats['Doubles'])
        doubles_pct = float(player_stats['Doubles %'])

        if outs_remaining <= 0:
            break

        if at_bats == 0:
            continue

        # Calculate probability of double based on doubles percentage
        double_probability = doubles_pct / 100

        # Determine number of doubles expected based on hits and double percentage
        expected_doubles = hits * double_probability

        # Calculate expected runs for player
        expected_runs = hits + (expected_doubles * 2)

        # Determine remaining number of outs after this player's at bats
        outs_remaining -= at_bats - hits

        # Update runs scored based on expected runs and remaining outs
        if outs_remaining >= 0:
            runs += expected_runs

    return runs


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
    # Get the players' doubles probability and average
    players = processed_data.set_index('Name')
    doubles_prob = players['Doubles %'].to_dict()
    batting_avg = players['Average'].to_dict()
    # Create a dictionary to hold the lineup order and their probabilities of getting a double
    lineup_with_prob = {
        player: doubles_prob[player] for player in optimal_order}
    # Sort the lineup in descending order of their probability of getting a double
    sorted_lineup = sorted(lineup_with_prob.items(),
                           key=lambda x: x[1], reverse=True)
    # Create a dictionary to hold the lineup order and their batting averages
    lineup_with_avg = {player: batting_avg[player] for player in optimal_order}
    # Sort the lineup in descending order of their batting average
    sorted_lineup_avg = sorted(
        lineup_with_avg.items(), key=lambda x: x[1], reverse=True)
    # Create a list of player names ordered by probability of getting a double and then batting average
    final_lineup = [player[0] for player in sorted_lineup]
    remaining_outs = 15
    inning_score = 0
    inning_outs = 0
    for i in range(len(final_lineup)):
        player_name = final_lineup[i]
        if inning_outs < 3 and remaining_outs > 0:
            at_bats = processed_data.loc[processed_data['Name']
                                         == player_name, 'At_Bats'].iloc[0]
            doubles = processed_data.loc[processed_data['Name']
                                         == player_name, 'Doubles'].iloc[0]
            probability = doubles_prob[player_name]
            runs_scored = 0
            for j in range(at_bats):
                if random.random() < probability:
                    runs_scored += 2
                else:
                    runs_scored += 1
                remaining_outs -= 1
                inning_outs += 1
                if remaining_outs == 0 or inning_outs == 3:
                    break
            inning_score += runs_scored
        else:
            break
    return jsonify({'optimal_batting_order': final_lineup, 'inning_score': inning_score})


if __name__ == '__main__':
    app.run(debug=True)
