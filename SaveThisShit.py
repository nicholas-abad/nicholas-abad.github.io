import urllib.request
import pandas as pd
import csv
import os

original_path = "/Users/Nick/Desktop/DataMining_Pictures/"
name_of_csv = "/Users/Nick/Desktop/analyze.csv"

data_frame = pd.read_csv(name_of_csv, sep = ',', header = None)


testing_pics_per_year = 1
testing_pics_per_decade = testing_pics_per_year * 10


training_pics_per_year = 10
training_pics_per_decade = training_pics_per_year * 10

amount_of_decades = 5

total_amount_of_pictures = (testing_pics_per_decade + training_pics_per_decade) * amount_of_decades


URL = []
decade = []
year = []


for i in range(1, total_amount_of_pictures + 1):
		decade.append(data_frame[1][i])
		year.append(data_frame[2][i])
		URL.append(data_frame[3][i])




URL_training = []
URL_testing = []

decade_training = []
decade_testing = []

year_training = []
year_testing = []

for i in range((testing_pics_per_year + training_pics_per_year) * 10 * amount_of_decades):
	if (i % (testing_pics_per_year + training_pics_per_year) != training_pics_per_year):
		URL_training.append(URL[i])
		decade_training.append(decade[i])
		year_training.append(year[i])
	else:
		URL_testing.append(URL[i])
		decade_testing.append(decade[i])
		year_testing.append(year[i])



# Training
for i in range(len(URL_training)):
	image = URL_training[i]
	new_path = original_path + str("Training/") + str(decade_training[i]) + str("/") + str(year_training[i] + str("/"))
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	urllib.request.urlretrieve(image, new_path + str(i) + str(".jpg"))

# Testing 
for i in range(len(URL_testing)):
	image = URL_testing[i]
	new_path = original_path + str("Testing/") + str(decade_testing[i]) + str("/") + str(year_testing[i] + str("/"))
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	urllib.request.urlretrieve(image, new_path + str(i) + str(".jpg"))

