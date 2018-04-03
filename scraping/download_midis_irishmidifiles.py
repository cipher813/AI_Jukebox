from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import urllib.request
import shutil
import time

web_url = 'http://www.irishmidifiles.ie/irish%20midi%20files/'
abs_filepath = '/media/cipher000/DATA/Music/IrishMidiFiles/'

def download_midis(web_url, abs_filepath):
    midi_dict = {}
    response = requests.get(web_url)
    page = response.text
    soup = BeautifulSoup(page,'lxml')
    tables=soup.find_all("pre")
    rows=[row for row in tables[0].find_all('a')]

    file_list = []
    for r in rows:
        r = str(r).lower()
        if '.mid"' in r:
            r = r.split('/')[2].split('"')[0]
            file_list.append(r)

    for f in file_list:
        url = web_url + f
        name = re.sub(r"[%?\d )(-/';:]",'',f.replace('.mid',''))
        filepath = abs_filepath + name + '.mid'

        with urllib.request.urlopen(url) as response, open(filepath, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        print("Midi at URL {} saved as {}".format(url,filepath))

        midi_dict[name] = url
    #
    #     except Exception as error:
    #         print(error)
    #         pass
    #     sleep_time = 60*5
    #     print("Sleeping {} seconds".format(sleep_time))
    #     time.sleep(sleep_time)
    # return midi_dict
#
midi_dict = download_midis(web_url, abs_filepath)
# print("Scraping Complete.")
# print(midi_dict)
