import json
import math
import os
import pickle
import random
import time
from .neural_network import NeuralNetwork


class Agent:
	def __init__(self, model, average_fitness, games_played):
		self.model = model
		self.average_fitness = average_fitness
		self.games_played = games_played
	
	def deep_clone(self):
		return Agent(self.model.deep_clone(), self.average_fitness, self.games_played)

class AdversarialGeneticAlgorithm:
	# dimensions: list of ints
	# population_size: int
	# input_func: game state -> neural network input array
	# run_game_func: agent1, agent2, input_func -> final game state
	# fitness_func: final game state -> two element list of fitness of both agents
	def __init__(self, dimensions, population_size, num_opponents, rounds_per_opponent,
							 input_func, run_game_func, fitness_func, save_agents=False):
		self.dimensions = dimensions
		self.population_size = population_size
		self.num_opponents = num_opponents
		self.rounds_per_opponent = rounds_per_opponent
		self.input_func = input_func
		self.run_game_func = run_game_func
		self.fitness_func = fitness_func
		self.gen_count = 0
		self.population = []
		self.best_agent = None
		self.save_agents = save_agents
		self.start_time = int(time.time())

	def run_generation(self):
		if self.gen_count == 0:
			# Generate initial population pool
			for _ in range(self.population_size):
				self.population.append(Agent(NeuralNetwork(self.dimensions), 0, 0))

		# Reset all old fitness
		for agent in self.population:
			agent.average_fitness, agent.games_played = 0, 0

		best_agents = self.population[:self.num_opponents]
		cloned_best_agents = [agent.deep_clone() for agent in best_agents]

		for agent in self.population:
			net = agent.model
			# Run the process and evaluate fitness
			# Everyone plays vs the top agents in the pool, except self
			for opponent_agent in best_agents:
				opponent_net = opponent_agent.model
				if net is opponent_net:
					continue

				avg_net_fitness = 0
				avg_opponent_fitness = 0
				for _ in range(self.rounds_per_opponent):
					final_game_state = self.run_game_func(net, opponent_net, self.input_func)
					net_fitness, opponent_fitness = self.fitness_func(final_game_state)
					avg_net_fitness += net_fitness
					avg_opponent_fitness += opponent_fitness
				avg_net_fitness /= self.rounds_per_opponent
				avg_opponent_fitness /= self.rounds_per_opponent

				def update_agent_fitness(ag, next_avg):
					old_avg = ag.average_fitness
					old_n = ag.games_played

					ag.games_played += self.rounds_per_opponent
					ag.average_fitness = ((old_avg * old_n) + (next_avg * self.rounds_per_opponent)) / ag.games_played
				
				update_agent_fitness(agent, avg_net_fitness)
				update_agent_fitness(opponent_agent, avg_opponent_fitness)

		# Select / merge
		# Sort by fitness, descending order
		self.population.sort(key=lambda p: p.average_fitness, reverse=True)
		self.best_agent = self.population[0]

		# Save best agent to disk
		if not os.path.exists('agents/'):
			os.mkdir('agents/')
		with open(f'agents/{self.start_time}_{self.gen_count}', 'wb') as file:
			pickle.dump(self.best_agent.model, file)

		# Randomly pick from pop -> best more likely
		def ran_idx():
			return int(4 * (-math.log(-random.random() + 1) + 0.1))

		new_population = []
		for _ in range(len(self.population) - self.num_opponents):
			i1 = ran_idx()
			i2 = ran_idx()
			if i1 >= len(self.population):
				i1 = 0
			if i2 >= len(self.population):
				i2 = 0
			merged = self.population[i1].model.merge(self.population[i2].model)
			new_population.append(Agent(merged, 0, 0))

		# Mutate
		for agent in new_population:
			agent.model.mutate()

		# Add best agents from prev generation unmutated
		new_population.extend(cloned_best_agents)

		self.population.clear()
		self.population.extend(new_population)

		self.gen_count += 1

	def get_best_agent(self):
		return self.best_agent
