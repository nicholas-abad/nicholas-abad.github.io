import requests
import string
import re
import os

from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin
from lxml import html
from tqdm import tqdm


class Song():
	""" 
	This class is used to scrape the lyrics of a single song given the title, artist name, and www.MetroLyrics.com URL
	"""


	def __init__(self, title = None, artist = None, url = None):
		self.lyrics = []
		self.title = title.replace(' ', '_').replace('.','_').lower()
		self.artist = artist.replace(' ', '_').replace('.','_').lower()
		self.url = url

	def load_lyrics(self):
		"""
		Loads lyrics of a specific song into self.lyrics, which is defined to be a list
		"""
		req = requests.get(self.url)
		soup = BeautifulSoup(req.text, "html.parser")
		main = soup.find('div', {'id': 'lyrics-body-text'})
		for verse in main.find_all('p', {'class': 'verse'}):
			self.lyrics.append(verse.text.strip())
		self.lyrics = '\n\n'.join(self.lyrics)

	def write_lyrics(self):
		"""
		Given that the function load_lyrics() has already been called, this function writes self.lyrics into a .txt file
		The lyrics are written within a directory called "Rapper" and furthermore written into a subdirectory of the rappers name
		If the directories are not yet created, this function will do so for you.
		"""
		self.load_lyrics()
		filename = self.title + str(".txt")
		directory = os.getcwd() + "/Rappers/{}/".format(self.artist)
		if not os.path.exists(directory):
			os.makedirs(directory)
		whole = os.path.join(directory + filename)
		outfile = open(whole, 'w')
		outfile.write(self.lyrics)

class Artist():
	"""
	This class is used in conjunction with the Song class in order to load all of the artist's song lyrics
	"""
	def __init__(self, name, artist_url):
		self.songs = []
		self.original_name = name
		self.name = name.replace(' ', '-').lower()
		self.artist_url = artist_url
		self.song_title = []

	def load_songs(self):
	"""
	For all the songs of an artist, the lyrics of each song are added into the list self.songs
	Additionally, the title of that song will be added to the list self.song_title
	
	For example, say that a song called "Example Song" is the first song that is chosen
	The lyrics of "Example Song" will be appended specifically at self.songs[0]
	Likewise, the title "Example Song" will be appended specifically at self.song_title[0]
	This process is iterated until all the songs within the artists' www.MetroLyrics.com page is completed
	"""
		page_num = 1
		total_pages = self.getNumPages()
		print("TOTAL PAGES: {}".format(total_pages))
		while page_num <= total_pages:
			page = requests.get("http://www.metrolyrics.com/{artist}-alpage-{n}.html".format(artist = self.name, n = page_num))
			tree = html.fromstring(page.text)
			song_rows_xp = r'//*[@id="popular"]/div/table/tbody/tr'
			songlist_pagination_xp = r'//*[@id="main-content"]/div[1]/'\
			'div[2]/p/span/a'
			rows = tree.xpath(song_rows_xp)
			for row in rows:
				song_link = row.xpath(r'./td/a[contains(@class,"title")]')
				assert len(song_link) == 1
				self.song_title.append(song_link[0].attrib['title'].replace(self.original_name + " ", "").replace(" ", "_").replace("_lyrics", "").replace("/","").lower())
				self.songs.append(song_link[0].attrib['href'])
			page_num += 1

	def getNumPages(self):
	"""
	This function returns the amount of pages of songs that the scraper needs to load in on www.MetroLyrics.com
	"""
		req = requests.get(self.artist_url)
		soup = BeautifulSoup(req.text, "html.parser")
		main = soup.find('span', {'class': 'pages'})
		return(len(main.find_all({'a' : 'href'})))

	def writeAllLyrics(self):
	"""
	After the function load_songs() is called, this function writes all the song lyrics into a specific file
	In order to do this, this function calls the Song class and from there calls the Song.write_lyrics() method
	"""
		self.load_songs()
		for i in tqdm(range(len(self.songs))):
			newSong = Song(title = self.song_title[i], artist = self.original_name, url = self.songs[i])
			newSong.write_lyrics()

# Case 1: Kanye West and Jay Z
hova = Artist("Jay_Z", "http://www.metrolyrics.com/jay-z-alpage-1.html")
yeezy = Artist("Kanye West", "http://www.metrolyrics.com/kanye-west-alpage-1.html")

# Case 2: Mos Def and Talib Kweli
mos = Artist("Mos Def", "http://www.metrolyrics.com/mos-def-alpage-1.html")
talib = Artist("Talib Kweli", "http://www.metrolyrics.com/talib-kweli-alpage-1.html")

# Case 3: Drake and Future
drake = Artist("Drake", "http://www.metrolyrics.com/drake-alpage-1.html")
future = Artist("Future", "http://www.metrolyrics.com/future-alpage-1.html")


mos.writeAllLyrics()
talib.writeAllLyrics()
hova.writeAllLyrics()
yeezy.writeAllLyrics()
drake.writeAllLyrics()
future.writeAllLyrics()






