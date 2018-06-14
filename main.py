import connect4
import numpy as np
import random
import math
import time
import copy

from anytree import Node, RenderTree
from tqdm import tqdm
import pyexcel as pe

random.seed(5)

class newAgent():
	def __init__(self, name, num_simulations, constant, actual_board):
		self.name = name
		self.num_simulations = num_simulations
		self.board = actual_board
		self.root = Node(name = "starting", board = self.board, parent = None, action = None, reward = 0, times_visited = 0, discovered = True, fully_explored = False)
		self.constant = constant

		self.simulation_board = copy.deepcopy(actual_board)
		self.turn = name

	def clear_simulation_board(self):
		self.simulation_board = copy.deepcopy(self.board)

	def random_move_in_simulation(self, player_to_move, simBoard):
		# print(simBoard.current_board)
		possible_actions = simBoard.legalMoves()
		if len(possible_actions) == 0:
			simBoard.winner = True
		else:
			random_action = random.choice(possible_actions)
			height = simBoard.getColumnHeight(random_action)
			simBoard.current_board[height][random_action] = player_to_move
			simBoard.checkForWinner()

	def run_single_random_simulation(self, node_to_run_simulation_on):
		simulation_turn = self.turn * -1

		copied = copy.deepcopy(node_to_run_simulation_on)
		# Testing a single simulation

		if copied.board.winner == False:
			while copied.board.winner == False:
				self.random_move_in_simulation(simulation_turn, copied.board)
				simulation_turn = simulation_turn * -1
		return(copied.board.winner)
			# Returns 1 if Player 1 wins
			# Returns 2 if Player 2 wins
			# Returns True if it's a tie

	def get_ucb(self, node):
		N = node.parent.times_visited
		if node.times_visited == 0:
			return(float('Inf'))

		else:
			exploration = node.reward / node.times_visited
			exploitation = self.constant * math.sqrt(math.log(N) / node.times_visited)
			return(exploration + exploitation)



	def selection(self, node_to_select_children_from):
		if len(node_to_select_children_from.board.legalMoves()) == 0:
			node_to_select_children_from.board.winner = 0
		else:
			possible_moves = node_to_select_children_from.board.legalMoves()

			# Case 1: No children yet but there are still moves it could make
			if (node_to_select_children_from.children == ()) and (len(possible_moves) != 0):
				random_action = random.choice(possible_moves)
				possible_moves.remove(random_action)

				if node_to_select_children_from.name == "starting":
					newNode = Node(name = self.turn, board = node_to_select_children_from.board.addAction(self.turn, random_action), parent = node_to_select_children_from, action = random_action, reward = 0, times_visited = 0, discoverd = True, fully_explored = False)
					winner_after_simulation = self.run_single_random_simulation(newNode)
					newNode.parent.times_visited += 1
					newNode.times_visited += 1

					if winner_after_simulation == self.name:
						newNode.parent.reward += 1
						newNode.reward += 1

					elif winner_after_simulation == self.name * -1:
						newNode.reward += -1
						newNode.parent.reward += -1

					self.clear_simulation_board()

				else:
					self.turn = node_to_select_children_from.name * -1
					newNode = Node(name = self.turn, board = node_to_select_children_from.board.addAction(self.turn, random_action), parent = node_to_select_children_from, action = random_action, reward = 0, times_visited = 0, discoverd = True, fully_explored = False)
					winner_after_simulation = self.run_single_random_simulation(newNode)
					if winner_after_simulation == self.name:
						currNode = newNode
						while currNode.parent != None:
							currNode.times_visited += 1

							currNode.reward += 1
							currNode = currNode.parent
						currNode.times_visited += 1

					elif winner_after_simulation == self.name * -1:
						currNode = newNode
						while currNode.parent != None:
							currNode.times_visited += 1

							currNode.reward += -1
							currNode = currNode.parent
						currNode.times_visited += 1


					self.clear_simulation_board()









			# Case 2: Has at least 1 child but there are still children that are undiscovered
			elif len(node_to_select_children_from.children) != len(possible_moves):			
				self.turn = self.turn
				# Populate a list called moves not played
				moves_not_played = possible_moves
				for i in node_to_select_children_from.children:
					if i.action in moves_not_played:
						moves_not_played.remove(i.action)


				# Add action from the remaining moves
				new_action_to_discover = random.choice(moves_not_played)
				if node_to_select_children_from.name == "starting":
					newNode = Node(name = self.turn, board = node_to_select_children_from.board.addAction(self.turn, new_action_to_discover), parent = node_to_select_children_from, action = new_action_to_discover, reward = 0, times_visited = 0, discoverd = True, fully_explored = False)
					winner_after_simulation = self.run_single_random_simulation(newNode)
					currNode = newNode
					while currNode.parent != None:
						currNode.times_visited += 1
						currNode = currNode.parent

					currNode.times_visited += 1
					# print("AGENT NAME: {}".format(self.name))
					if winner_after_simulation == self.name:
						newNode.parent.reward += 1
						newNode.reward += 1

					elif winner_after_simulation == self.name * -1:
						newNode.reward += -1
						newNode.parent.reward += -1

					self.clear_simulation_board()

				else:
					self.turn = node_to_select_children_from.name * -1
					newNode = Node(name = self.turn, board = node_to_select_children_from.board.addAction(self.turn, new_action_to_discover), parent = node_to_select_children_from, action = new_action_to_discover, reward = 0, times_visited = 0, discoverd = True, fully_explored = False)
					winner_after_simulation = self.run_single_random_simulation(newNode)
					if winner_after_simulation == self.name:
						currNode = newNode
						while currNode.parent != None:
							currNode.reward += 1
							currNode.times_visited += 1
							currNode = currNode.parent
						currNode.times_visited += 1

					elif winner_after_simulation == self.name * -1:
						currNode = newNode
						while currNode.parent != None:
							currNode.times_visited += 1
							currNode.reward += -1
							currNode = currNode.parent
						currNode.times_visited += 1





			# Case 3: Fully explored
			elif len(node_to_select_children_from.children) == len(possible_moves):
				node_to_select_children_from.fully_explored = True
				self.turn = self.turn * -1
				max_node = None
				max_ucb = float("-Inf")
				for child in node_to_select_children_from.children:
					if self.get_ucb(child) > max_ucb:
						max_ucb = self.get_ucb(child)
						max_node = child

				self.selection(max_node)








class mc():
	def __init__(self, current_board, player1_agent, player2_agent):
		self.current_board = current_board
		self.player1_agent = player1_agent
		self.player2_agent = player2_agent
		self.updateAgentsBoards()

	def updateAgentsBoards(self):
		self.player1_agent.board = self.current_board
		self.player2_agent.board = self.current_board

	def setEmpty(self):
		for row in range(len(self.current_board.current_board)):
			for entry in range(len(self.current_board.current_board[row])):
				self.current_board.current_board[row][entry] = 0
		self.current_board.winner = False

	def get_best_action(self, node):
		best_child = None
		best_ucb = float('-Inf')
		for children in node.children:
			if self.player1_agent.get_ucb(children) > best_ucb:
				best_ucb = self.player1_agent.get_ucb(children)
				best_child = children
		return(best_child)


	def player1_move(self):
		for i in range(self.player1_agent.num_simulations):
			self.player1_agent.selection(self.player1_agent.root)

		best_action = self.get_best_action(self.player1_agent.root)
		if type(best_action) != type(None):
			self.current_board = best_action.board
			self.player1_agent.root.board = self.current_board
			self.player2_agent.root.board = self.current_board
			self.player2_agent = newAgent(name = self.player2_agent.name, num_simulations = self.player2_agent.num_simulations, constant = self.player2_agent.constant, actual_board = self.current_board)
			self.updateAgentsBoards()
		else:
			self.current_board.winner = 0
			self.player1_agent.root.board = self.current_board
			self.player2_agent.root.board = self.current_board
			self.player2_agent = newAgent(name = self.player2_agent.name, num_simulations = self.player2_agent.num_simulations, constant = self.player2_agent.constant, actual_board = self.current_board)
			self.updateAgentsBoards()


	def player2_move(self):
		for i in range(self.player2_agent.num_simulations):
			self.player2_agent.selection(self.player2_agent.root)

		best_action = self.get_best_action(self.player2_agent.root)
		if type(best_action) != type(None):
			self.current_board = best_action.board
			self.player1_agent.root.board = self.current_board
			self.player2_agent.root.board = self.current_board
			self.player1_agent = newAgent(name = self.player1_agent.name, num_simulations = self.player1_agent.num_simulations, constant = self.player1_agent.constant, actual_board = self.current_board)

			self.updateAgentsBoards()
		else:
			self.current_board.winner = 0
			self.player1_agent.root.board = self.current_board
			self.player2_agent.root.board = self.current_board
			self.player2_agent = newAgent(name = self.player2_agent.name, num_simulations = self.player2_agent.num_simulations, constant = self.player2_agent.constant, actual_board = self.current_board)
			self.updateAgentsBoards()

	def check_if_draw(self):
		if len(self.current_board.legalMoves()) == 0:
			self.current_board.winner = 0
			return(True)
		else:
			return(False)

	def run_single_game(self):
		while self.current_board.winner == False:
			self.player1_move()
			self.current_board.checkForWinner()
			self.player2_move()
			self.current_board.checkForWinner()
			if self.check_if_draw() == True:
				self.current_board.winner = 0
				break

		single_winner = self.current_board.winner
		return(single_winner)


	def run_multiple_games(self, num_games):
		file =[]
		headers = ['num_games', 'player 1 wins', 'player 2 wins', 'draws', 'player 1 simulations', 'player 2 simulations', 'player 1 constant', 'player 2 constant']
		file.append(headers)


		player1_constants = [.9, 1.0, 1.1]
		for constant in player1_constants:
			player1_wins = 0
			player2_wins = 0
			draws = 0
			self.player1_agent.constant = constant
			new_game = []
			for i in tqdm(range(num_games)):
				winner = self.run_single_game()
				if winner == 1:
					player1_wins += 1
				elif winner == -1:
					player2_wins += 1
				else:
					draws += 1
				print(self.current_board.current_board)
				self.setEmpty()
			new_game = [num_games, player1_wins, player2_wins, draws, self.player1_agent.num_simulations, self.player2_agent.num_simulations, self.player1_agent.constant, self.player2_agent.constant]
			file.append(new_game)

		sheet = pe.Sheet(file)
		sheet.save_as("testing_optimal_constants.csv")





	def test(self):		
		self.player1_move()
		self.player1_move()
		print(self.player1_agent.root)
		print()
		for i in self.player1_agent.root.children:
			print(i, self.player2_agent.get_ucb(i))






np_board = np.zeros(shape = [7, 7])

board = connect4.updatedBoard(np_board, 4)
player1 = newAgent(name = 1, num_simulations = 50, constant = 1.0, actual_board = board)
player2 = newAgent(name = -1, num_simulations = 50, constant = 1.3, actual_board = board)

mc = mc(board, player1, player2)
mc.run_multiple_games(2)




















