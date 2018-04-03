from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import urllib.request
import shutil
import time

base_url = 'http://www.tadpoletunes.com/tunes/'
end_url = '.htm'
abs_filepath = '/media/cipher000/DATA/Music/Tadpole/Celtic/'

slug_list = ['celtic1/celtic','celtic2/celtic2','celtic3/celtic3']

def download_midis(slug_list,base_url, end_url, abs_filepath):
    midi_dict = {}
    for slug in slug_list:
        url = base_url + slug + end_url
        # print(url)
        try:
            response = requests.get(url)
            page = response.text
            soup = BeautifulSoup(page,'lxml')
            tables=soup.find_all("table")
            rows=[row for row in tables[2].find_all('a')]
            # print(rows)

            name_list = []
            midi_list = []

            for row in rows:
                if '.mid' in str(row):
                    r = str(row)
                    midi = r.split('"')[1]
                    loc1 = r.find('b>')
                    loc2 = r.find('</')
                    name = r[loc1+2:loc2]
                    # print("Midi: {} Name: {}".format(midi,name))

                    name_list.append(name)
                    midi_list.append(midi)

            slug = slug.split('/')[0] + '/'
            for i in range(len(midi_list)):

                name = name_list[i]
                midi = midi_list[i]
                song = re.sub(r"[?\d )(-/';:]",'',midi.replace('.mid','').replace('\n',''))
                name = re.sub(r"[? )(-/';:]",'',name.replace('\n',''))
                url_path = base_url + slug + midi
                # print("{}-{}-{}".format(name,song,url_path))

                filepath = abs_filepath + "{}.mid".format(song)

                with urllib.request.urlopen(url_path) as response, open(filepath, 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)

                print("Midi at URL {} saved as {}".format(url_path,filepath))

                midi_dict[song] = {'Name':name, 'URL': url_path}

        except Exception as error:
            print(error)
            pass
        sleep_time = 60*5
        print("Sleeping {} seconds".format(sleep_time))
        time.sleep(sleep_time)
    return midi_dict
#
midi_dict = download_midis(slug_list, base_url, end_url, abs_filepath)
print("Scraping Complete.")
print(midi_dict)
