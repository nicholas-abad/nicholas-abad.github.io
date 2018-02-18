# Scrape indeed.com for data science jobs and key words?
import json
import string
import requests
from urllib.parse import urljoin #https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urljoin
from bs4 import BeautifulSoup, Tag #https://www.crummy.com/software/BeautifulSoup/bs4/doc/
import re
import csv
from tqdm import tqdm
import pyexcel as pe






base_url = "https://www.indeed.com/jobs?q=data+scientist&l=San+Francisco&fromage=last&sort=date"


req = requests.get(base_url)
soup = BeautifulSoup(req.text, "html.parser")
main = soup.find('td', {'id' : 'resultsCol'})

def getJobTitlesAndURL(main = main):
	job_list = []
	for jobs in main.find_all('div', {'class': re.compile(r"row result")}):
		company = jobs.find('span', {'class': 'company'}).text.strip()
		location = jobs.find('span', {'class': 'location'}).text.strip()
		title = jobs.find('h2', {'class': 'jobtitle'}).text.strip()
		age = jobs.find('span', {'class': 'date'}).text.strip()
		if company in job_list:
			company = company + "*"
		if re.search("[Dd]ata [Ss]ci", title):
			if re.search("^((?![Ss]enior).)*$", title) and re.search("^((?![Mm]anage).)*$", title):
				if re.search("^((?![Pp][Hh]d).)*$", title) and re.search("^((?![Ll]ead).)*$", title):
					if re.search("^((?![Dd]irector).)*$", title) and re.search("^((?![Pp]rincipal).)*$", title):
						url = "https://www.indeed.com" + str(jobs.find('h2', {'class': 'jobtitle'}).a['href'])
						job_list.append({'company': company, 'title': title, 'location': location, 'age': age, 'url': url})

	return(job_list)




# numPages is the amount of pages at the bottom
def getAllJobTitles(numPages = 10):
	original_url = 'https://www.indeed.com/jobs?q=data+scientist&l=San+Francisco&sort=date&fromage=last&start='
	allJobs = []
	for i in tqdm(range(0, numPages)):
		newPage = original_url + str(i * 10)
		newReq = requests.get(newPage)
		newSoup = BeautifulSoup(newReq.text, "html.parser")
		newPage_main = newSoup.find('td', {'id' : 'resultsCol'})
		# Dictionary's version of append is .update
		list = getJobTitlesAndURL(newPage_main)
		for i in list:
			allJobs.append(i)

	return(allJobs)

def createJSON(dictionary, output_name):
    with open(output_name, 'w') as f:
        json.dump(dictionary, f, indent=4)

def createCSV(json_file, output_file):
	with open(json_file) as json_data:
	    d = json.load(json_data)
	    json_data.close()
	allJobs = []
	headers = ['Company Name', 'Job Title', 'Location', 'Date Posted', 'URL']
	allJobs.append(headers)
	for i in range(len(d)):
		newJob = []
		newJob.append(d[i]['company'])
		newJob.append(d[i]['title'])
		newJob.append(d[i]['location'])
		newJob.append(d[i]['age'])
		newJob.append(d[i]['url'])
		allJobs.append(newJob)

	sheet = pe.Sheet(allJobs)
	sheet.save_as(output_file)



createJSON(getAllJobTitles(30), "Indeed_jobs.json")
createCSV("Indeed_jobs.json", "Indeed_jobs.csv")

