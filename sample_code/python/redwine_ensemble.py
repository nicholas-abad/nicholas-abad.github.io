import math
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
import pyexcel as pe
import time

# Note that the first 11 columns are the explanatory variables
# The last column is thus the binary response variable
df = pd.read_csv("redwine_nolabels.csv", header = 0)
df.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
X = df[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
X = np.array(X)

# Changes every explanatory variable value to fit between -1 and 1
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
X = min_max_scaler.fit_transform(X)
Y = np.array(df[['quality']])

# Splits the whole dataset into a training and testing set using a 80/20 split
X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size = 0.20)

# Returns the logistic regression probability
def predict(betas, observation):
	t = 0
	for i in range(len(betas)):
		t += observation[i] * betas[i]
	prob = (1 / (1 + math.exp(-t)))
	return(prob)

# Returns the updated cost due to each error
def getCost(X_train, Y_train , betas, obs_val, learning_rate):
	total = 0
	amount_of_observations = len(Y_train)
	for i in range(amount_of_observations):
		individual_value = X_train[i][obs_val]
		prediction = predict(betas,X_train[i])
		error = individual_value * (prediction - Y_train[i])
		total += error
	# If you leave it as just the learning_rate, then there's a math overflow problem in Python
	learning_rate = float(learning_rate)/float(amount_of_observations)
	update = learning_rate * total
	return(update)

# By subtracting the costs from the getCost() function from the preexisting betas, this simply updates the new betas
# Returns the new betas
def updateBetas(X_train, Y_train , old_betas, learning_rate):
	new_betas = []
	for i in range(len(old_betas)):
		newBetas = old_betas[i] - getCost(X_train, Y_train , old_betas, i, learning_rate)
		new_betas.append(newBetas)
	return(new_betas)

# Using the previous functions, this firstly trains the logistic model on the inputted training set to get the predicted beta values
# With these beta values that were just calculated, the testing observations are then inputted into the predict() function to output a probability
# With the probabilities for each of the testing observations, they are each rounded to the nearest whole number (either 0 or 1)
	# This is one of the items returned and is called 'predictions' within this function
# As more of a help to me throughout the process, this also returned the percentage correct and the percentage incorrect
def getProbabilities(X_train, Y_train, X_test, Y_test, learning_rate, betas, iterations):
	for x in range(iterations):
		new_beta = updateBetas(X_train,Y_train,betas,learning_rate)
		betas = new_beta
	numCorrect = 0
	numWrong = 0
	total_observations = len(Y_test)
	predictions = []
	for i in range(total_observations):
		prediction = round(predict(betas, X_test[i]))
		predictions.append(predict(betas, X_test[i]))
		actual = Y_test[i]
		if prediction == actual:
			numCorrect += 1
		else:
			numWrong += 1
	percentageCorrect = numCorrect/total_observations
	percentageWrong = numWrong/total_observations
	# print("Num correct: " + str(numCorrect) + " ( " + str(percentageCorrect) + "% )")
	# print("Num wrong: " + str(numWrong) + " ( " + str(percentageWrong) + "% )")

	return(predictions, percentageCorrect, percentageWrong)

# This was used to initialize the betas
# If the parameter 'initialize_at_zero' == True, each beta value is intuitively initialized at 0
# Otherwise, each beta value takes a random value in between 0 and 1
# An array with these betas is then returned
def getInitialBetas(colNames, initialize_at_zero = True):
	initial_betas = []
	if initialize_at_zero == False:
		for i in range(len(colNames) - 1):
			# The "-1" is for the last variable which is the response
			initial_betas.append(np.random.uniform(0,1))
	else:
		for i in range(len(colNames) - 1):
			initial_betas.append(0)

	return(initial_betas)

# This was more of a helper function
# If given an array with multiple probabilities, it just returns a prediction array
# If a probability is less than a threshold (which was initialized to be 0.5 but can be changed), the prediction array appends a 0
# Otherwise, the prediction array appends a 1
# This prediction array is then returned
def getPredictions(probability_array, threshold = 0.5):
	predictions = []
	for i in probability_array:
		if i < threshold:
			predictions.append(0)
		else:
			predictions.append(1)
	return(predictions)

# Given a prediction array constructed in getPredictions(), this is compared to the actual values of the testing set
# This function thus calculates and returns the accuracy, precision and recall values
def getAccuracyMeasures(prediction_array, Y_test):
	# True positive: predicted to be a 1 and is actually a 1
	true_positive = 0
	# True negative: predicted to be a 0 and is actually a 0
	true_negative = 0
	# False positive: predicted to be a 1 and is actually a 0
	false_positive = 0
	# False negative: predicted to be a 0 and is actually a 1
	false_negative = 0
	# print("LENGTH: " + str(len(prediction_array)) + " / " + str(len(Y_test)))
	for i in range(len(prediction_array)):
		if (prediction_array[i] and Y_test[i]) == 1:
			true_positive += 1
		elif (prediction_array[i] == 0 and Y_test[i] == 0):
			true_negative += 1
		elif (prediction_array[i] == 1 and Y_test[i] == 0):
			false_positive += 1
		elif (prediction_array[i] == 0 and Y_test[i] == 1):
			false_negative += 1
	if (true_negative + true_positive + false_negative + false_positive) != 0:
		accuracy = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)
	else:
		accuracy = -1

	if (true_positive + false_positive) != 0:
		precision = (true_positive) / (true_positive + false_positive)
	else:
		precision = -1

	if (true_positive + false_negative) != 0:
		recall = (true_positive) / (true_positive + false_negative)
	else:
		precision = -1

	return(accuracy, precision, recall)


class ensemble():
	# Initialization of the class
	def __init__(self, X_train, Y_train, percentage_of_training_used, numBags):
		self.whole_training = pd.DataFrame(np.hstack([X_train, Y_train]))
		self.length = len(self.whole_training)
		self.percentage_of_training_used = percentage_of_training_used
		self.numBags = numBags

	# Given the percentage of the training set that you want to use when training each logistic regression method, this first determines the AMOUNT of observations that will be needed to train the model
	# With this amount, this function simply calculates random integers between 0 and the length of the whole training set, which allows for repetition
		# Repetition is important in this case because we are supposed to be sampling with replacement
	# With these pseudo-random integers in hand, these will act as the indices of the observations in the original training dataset
	# We simply choose to populate these observations into a list and turn this into a pandas dataframe
	# With this pandas dataframe available, we are thus able to train our logistic regression model using these observations
	def getSingleBag(self):
		numObservations = round(self.length * self.percentage_of_training_used)
		indices_of_bagged_obs = []
		for i in range(numObservations):
			indices_of_bagged_obs.append(random.randint(0, self.length - 1))

		used_observations = []
		for i in range(len(indices_of_bagged_obs)):
			used_observations.append(self.whole_training.values[indices_of_bagged_obs[i]])

		pandas_dataframe = pd.DataFrame(used_observations)
		return(pandas_dataframe)

	# This function first gets a dataframe using the previous method getSingleBag()
	# This is used as the training set and is thus used to train the logistic regression model after creating the proper column names and transforming each value to be between -1 and 1
	# A probability array for a single test observation is thus returned using the getProbabilities() function
	def getSingleObsProbabilities(self, X_test, Y_test, learning_rate = 0.1, number_of_iterations = 10):
		training_set = self.getSingleBag()
		# print("TRAIN: " + str(training_set[0]))
		training_set.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
		initial_betas = getInitialBetas(training_set.columns)
		# print("LENGTH OF INITIAL BETAS: " + str(len(initial_betas)))
		new_X_train = training_set[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
		new_X_train = np.array(new_X_train)
		# print(new_X_train[0])
		new_X_train = min_max_scaler.fit_transform(new_X_train)
		new_Y_train = np.array(training_set[['quality']])
		probs, correct, wrong = getProbabilities(new_X_train, new_Y_train, X_test, Y_test, learning_rate, initial_betas, number_of_iterations)
		# print("LENGTH OF PROBS: " + str(len(probs)))
		return(probs, correct, wrong)

	# Once again, this is more of a helper method.
	# This is exactly the same as the function getPredictions(), which is located outside of the class
	# After thinking about it, I really don't need this function...
	def getPredictions(self, probability_array, threshold = 0.5):
		prediction_array = []
		for i in range(len(probability_array)):
			if probability_array[i] <= threshold:
				prediction_array.append(0)
			else:
				prediction_array.append(1)
		return(prediction_array)

	# This is where the ensemble actually takes place
	# The first thing it does is create a 'counter' array which initially has the same length as the number of test observations. Each value is 0.
	# Next, I iterate through the amount of logistic regression models that is specified in the initialization of the ensemble method
	# During the first iteration, I train an individual logistic regression model and get the probabilities using the getSingleObsProbabilities() method
	# This then iterates through that probability array and adds a 1 to the 'counter' array at counter[j] if that said probability is greater than the threshold
	# This is repeated as many times as numBags
	# Note that in each iteration, new observations are used to train the model so therefore, betas will be different in each case

	# This first portion of the code then has a counter array.
	#	Example: Say numBags = 5 and there are 5 test observations. A possible counter array could look like [3,4,5,1,3]
	#	In this example, for the first test observation, 3 of the 5 logistic regression models had probabilities greater than 0.5
	#	Likewise, for the test observation, 4 of the 5 logistic regression models had probabilities greeater than 0.5. 
	#	This keeps going

	# With this counter array, it is then divided by numBags and rounded to the nearest whole number, which is either 0 or 1
	# This will then be used as our ensembles prediction array
	def actualEnsemble(self, X_test, Y_test, learning_rate = 0.1, number_of_iterations = 7, threshold = 0.5):
		amount_of_test_observations = len(X_test)
		counter = [0] * amount_of_test_observations
		for i in range(self.numBags):
			probs, correct, wrong = self.getSingleObsProbabilities(X_test, Y_test, learning_rate, number_of_iterations)
			# print("PROBS LEN: " + str(len(probs)))
			# print("XTEST LEN: " + str(len(X_test)))
			# print()
			for j in range(len(probs)):
				if probs[j] > threshold:
					counter[j] += 1

		for i in range(len(counter)):
			counter[i] = round(counter[i] / self.numBags)

		return(counter)

percentage_used = 0.1
numModels = 5
ensemble = ensemble(X_train, Y_train, percentage_used, numModels)
counter = ensemble.actualEnsemble(X_test, Y_test)
accuracy, precision, recall = getAccuracyMeasures(counter, Y_test)
print()
print()
print("PREDICTIONS (1-10): " + str(counter[0:9]))
print("ACTUAL (1-10): " + str(Y_test.flatten()[0:9]))
print()
print("Accuracy for an ensemble: " + str(accuracy))
print("Precision for an ensemble: " + str(precision))
print("Recall for an ensemble: " + str(recall))
print()
print()



# Function used to vary through the amount of bags, the percentage of training observations used and the learning rates
# This outputs the accuracy, precision, and recall values to a .csv file, which I will then ultimately use in R
def varyBagsPercLR():
	csv_file = []
	dataset_names = ['numBags', 'percentage_of_training_used', 'learning_rate', 'number_of_iterations', 'accuracy', 'precision', 'recall', 'total_time']
	csv_file.append(dataset_names)
	perc = [0.1,.2,.3,.4,.5,.6,.7,.8,.9]

	for i in tqdm(range(1,12,2)):
		# this is percentage of training sets used
		for j in tqdm(perc):
			initialize_ensemble = time.time()
			test = ensemble(X_train, Y_train, j, i)
			ending_initialize = time.time() - initialize_ensemble
			for k in tqdm(range(1,1000,10)):
				initialize_array = time.time()
				prediction_array = test.actualEnsemble(X_test, Y_test, learning_rate = k/1000, number_of_iterations = 3, threshold = 0.5)
				accuracy, precision, recall = getAccuracyMeasures(prediction_array, Y_test)
				ending_array = time.time() - initialize_array
				total_time = ending_initialize + ending_array
				values = [i, j, k/1000, 3, accuracy, precision, recall, total_time]
				print("BAGS: " + str(i))
				print("PERC: " + str(j))
				print("LR: " + str(k/1000))
				csv_file.append(values)

		sheet = pe.Sheet(csv_file)
		sheet.save_as("varyBagsPercLR.csv")

# Function used to just vary through the amount of logistic regression models used in an ensemble
# This keep all other parameters constant
# This outputs the accuracy, precision, and recall values to a .csv file, which I will then ultimately use in R
def varyBags():
	percentage_of_training_used = 0.8
	accuracies = []
	header = ['numIterations', 'learning_rate', 'percentage_of_training_used', 'numBags', 'accuracy', 'precision', 'recall', 'time_in_seconds']
	accuracies.append(header)
	for i in tqdm(range(1,101,2)):
		start_time = time.time()
		numBags = i
		test = ensemble(X_train, Y_train, percentage_of_training_used, numBags)
		prediction_array = test.actualEnsemble(X_test, Y_test, learning_rate = 0.089, number_of_iterations = 3, threshold = 0.5)
		accuracy, precision, recall = getAccuracyMeasures(prediction_array, Y_test)
		time_taken = time.time() - start_time
		accuracies.append([3, 0.089, 0.8, i, accuracy, precision, recall, time_taken])

	sheet = pe.Sheet(accuracies)
	sheet.save_as("varyBags_report.csv")

# Function used to just vary through the learning rates when updating the beta values in a logistic regression model
# This keep all other parameters constant
# This outputs the accuracy, precision, and recall values to a .csv file, which I will then ultimately use in R
def varyLR():
	percentage_of_training_used = 0.8
	accuracies = []
	header = ['numIterations', 'learning_rate', 'percentage_of_training_used', 'numBags', 'accuracy', 'precision', 'recall', 'time_in_seconds']
	accuracies.append(header)
	for i in tqdm(range(1, 1000, 5	)):
		start_time = time.time()
		test = ensemble(X_train, Y_train, percentage_of_training_used, numBags = 3)
		prediction_array = test.actualEnsemble(X_test, Y_test, learning_rate = i/1000, number_of_iterations = 3, threshold = 0.5)
		accuracy, precision, recall = getAccuracyMeasures(prediction_array, Y_test)
		time_taken = time.time() - start_time
		accuracies.append([5, i/1000, 0.8, 3, accuracy, precision, recall, time_taken])

	sheet = pe.Sheet(accuracies)
	sheet.save_as("varyLR.csv")

# Function used to just vary the amount of time iterations within the logistic regression model are calculated
# This keep all other parameters constant
# This outputs the accuracy, precision, and recall values to a .csv file, which I will then ultimately use in R
def varyIters():
	percentage_of_training_used = 0.8
	accuracies = []
	header = ['numIterations', 'learning_rate', 'percentage_of_training_used', 'numBags', 'accuracy', 'precision', 'recall', 'time_in_seconds']
	accuracies.append(header)
	for i in tqdm(range(1, 26)):
		start_time = time.time()
		test = ensemble(X_train, Y_train, percentage_of_training_used, numBags = 3)
		prediction_array = test.actualEnsemble(X_test, Y_test, learning_rate = 0.089, number_of_iterations = i, threshold = 0.5)
		accuracy, precision, recall = getAccuracyMeasures(prediction_array, Y_test)
		time_taken = time.time() - start_time
		accuracies.append([i, 0.089, 0.8, 3, accuracy, precision, recall, time_taken])

	sheet = pe.Sheet(accuracies)
	sheet.save_as("varyIters.csv")

# Function used to just vary through the percentage of the original training set that is used when training the logistic regression models within the ensemble
# This keep all other parameters constant
# This outputs the accuracy, precision, and recall values to a .csv file, which I will then ultimately use in R
def varyPerc():
	percs = [0.1,0.15, 0.2,0.25, 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
	accuracies = []
	header = ['numIterations', 'learning_rate', 'percentage_of_training_used', 'numBags', 'accuracy', 'precision', 'recall', 'time_in_seconds']
	accuracies.append(header)
	for i in percs:
		start_time = time.time()
		test = ensemble(X_train, Y_train, i, numBags = 3)
		prediction_array = test.actualEnsemble(X_test, Y_test, learning_rate = 0.089, number_of_iterations = 10, threshold = 0.5)
		accuracy, precision, recall = getAccuracyMeasures(prediction_array, Y_test)
		time_taken = time.time() - start_time
		accuracies.append([10, 0.089, i, 3, accuracy, precision, recall, time_taken])

	sheet = pe.Sheet(accuracies)
	sheet.save_as("varyPerc.csv")






