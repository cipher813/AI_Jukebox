from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import urllib.request
import time

base_url = 'http://www.midiworld.com/search/'
end_url = '/?q=movie%20themes'
abs_filepath = '/media/cipher000/DATA/Music/MidiWorld/Movie/'

start_page = 1
end_page = 5

def download_midis(start_page, end_page, base_url, end_url, abs_filepath):
    midi_dict = {}
    for num in range(start_page,end_page):
        url = base_url + str(num) + end_url
        print(url)
        try:
            response = requests.get(url)
            page = response.text
            soup = BeautifulSoup(page,'lxml')
            name_list = []

            for e in soup.find_all('li'):
                if 'download' in str(e):
                    e = str(e)
                    beg_loc = e.find(' ')
                    end_loc = e.find(' - ')
                    name = e[beg_loc:end_loc]
                    name_list.append(name)
            # print("Name list: {}".format(name_list))
            url_list = []

            midi = soup.find_all('a',target='_blank')

            for link in midi:
                url = link.get('href')
                url_list.append(url)

            # print("URL list: {}".format(url_list))
            for i in range(len(name_list)):
                name_text = name_list[i]
                mid_loc = name_text.find('(')
                end_loc = name_text.find(')')
                song = name_text[:mid_loc]
                artist = name_text[mid_loc+1:end_loc]
                url = url_list[i]
                song = re.sub(r"[\d )(-/';:]",'',song)
                artist = re.sub(r"[ )(-/';:]",'',artist)
                filepath = abs_filepath + "{}-{}.mid".format(artist,song)
                # print("Saving midi to {}".format(filepath))
                with urllib.request.urlopen(url) as response, open(filepath, 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)

                # urllib.request.urlretrieve(url, filepath)
                # r = requests.get(url)

                midi_dict[song] = {'Artist':artist, 'URL': url}

                print("{}-{}-{}".format(artist,song,url)) # song, artist, url
        except Exception as e:
            print(e)
            pass
        sleep_time = 60*5
        print("Sleeping {} seconds".format(sleep_time))
        time.sleep(sleep_time)
    return midi_dict

midi_dict = download_midis(start_page, end_page, base_url, end_url, abs_filepath)
print("Scraping Complete.")
print(midi_dict)
