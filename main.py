from os import listdir
from os.path import isfile, join, splitext, split
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import string
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import pandas as pd

def whitespace_tokenizer(text):
	"""
	Tokeniser that will be used
	"""
	return(text.split(" "))

def extract_text(folder):
	"""
	Extracts the contents within .txt files within a specific folder and adds them to a list
	This list is then returned
	"""
	textfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".txt")]
	texts = []
	for tf in textfiles:
		with open(tf) as f:
			text = f.read()
			texts.append(text)
	return texts

def ensemble(predictions1, predictions2, predictions3):
	"""
	Given three lists of predictions, this ensemble method returns the prediction that a majority of the predictions choose
	This utilises a simple voting based method when choosing the best prediction
	"""
	final_predictions = []
	for i in range(len(predictions1)):
		if predictions1[i] == predictions2[i]:
			final_predictions.append(predictions1[i])
		elif predictions1[i] == predictions3[i]:
			final_predictions.append(predictions1[i])
		else:
			final_predictions.append(predictions2[i])
	return(final_predictions)


if __name__ == '__main__':
	## Case 1: Watch The Throne
	artist1 = "jay-z"
	artist2 = "kanye_west"


	#### UNCOMMENT FOR CASE 2
	## Case 2: Blackstar
	# artist1 = "mos_def"
	# artist2 = "talib_kweli"

	#### UNCOMMENT FOR CASE 3
	## Case 3: What A Time To Be Alive
	# artist1 = "drake"
	# artist2 = "future"

	artist1_train_lyrics = extract_text("Rappers/{}".format(artist1))
	artist2_train_lyrics = extract_text("Rappers/{}".format(artist2))
	train_texts = artist1_train_lyrics + artist2_train_lyrics
	train_labels = ["{}".format(artist1)] * len(artist1_train_lyrics) + ["{}".format(artist2)] * len(artist2_train_lyrics)


	artist1_test_lyrics = extract_text("joint/{}".format(artist1))
	artist2_test_lyrics = extract_text("joint/{}".format(artist2))
	test_texts = artist1_test_lyrics + artist2_test_lyrics
	test_labels = ["{}".format(artist1)] * len(artist1_test_lyrics) + ["{}".format(artist2)] * len(artist2_test_lyrics)

	logistic_pipeline = Pipeline([
		('vect', CountVectorizer(analyzer='word', stop_words=stopwords.words('english') + list(string.punctuation), tokenizer = whitespace_tokenizer, max_features=500)),
		('tfidf', TfidfTransformer()),  
		('clf', LogisticRegression())
		])

	svm_pipeline = Pipeline([
		('vect', CountVectorizer(analyzer='word', stop_words=stopwords.words('english') + list(string.punctuation), tokenizer = whitespace_tokenizer, max_features=500)),
		('tfidf', TfidfTransformer()),  
		('clf', LinearSVC())
		])

	rf_pipeline = Pipeline([
		('vect', CountVectorizer(analyzer='word', stop_words=stopwords.words('english') + list(string.punctuation), tokenizer = whitespace_tokenizer, max_features=500)),
		('tfidf', TfidfTransformer()),  
		('clf', RandomForestClassifier())
		])

	# Logistic Regression
	logistic_pipeline.fit(train_texts, train_labels)
	logistic_predictions = logistic_pipeline.predict(test_texts)
	logistic_probabilities = logistic_pipeline.predict_proba(test_texts)

	# Support Vector Machines
	svm_pipeline.fit(train_texts, train_labels)
	svm_predictions = svm_pipeline.predict(test_texts)

	# Random Forests
	rf_pipeline.fit(train_texts, train_labels)
	rf_predictions = rf_pipeline.predict(test_texts)
	rf_probabilities = rf_pipeline.predict_proba(test_texts)

	# Get the final predictions using the ensemble() function
	final_predictions = ensemble(logistic_predictions, svm_predictions, rf_predictions)



	# Initialise the number correct
	logistic_num_correct = 0
	svm_num_correct = 0
	rf_num_correct = 0

	# Increment if the test label is equal to the prediction
	for i in range(len(test_labels)):
		if test_labels[i] == logistic_predictions[i]:
			logistic_num_correct += 1
		if test_labels[i] == svm_predictions[i]:
			svm_num_correct += 1
		if test_labels[i] == rf_predictions[i]:
			rf_num_correct += 1

	# Create new variables that calculate the accuracy
	logistic_accuracy = logistic_num_correct / len(logistic_probabilities)
	svm_accuracy = svm_num_correct / len(svm_predictions)
	rf_accuracy = rf_num_correct / len(rf_predictions)
	print()
	print("Results for {} and {}".format(artist1, artist2))
	print(" 	Accuracy: {}".format(accuracy_score(test_labels, final_predictions)))
	print(" 	Precision: {}".format(np.mean(precision_score(test_labels, final_predictions, average = None))))
	print(" 	Recall: {}".format(np.mean(recall_score(test_labels, final_predictions, average = None))))
	print()
	print(" 	Logistic Regression Accuracy: {}".format(logistic_accuracy))
	print(" 	Support Vector Machine Accuracy: {}".format(svm_accuracy))
	print("	Random Forest Accuracy: {}".format(rf_accuracy))

