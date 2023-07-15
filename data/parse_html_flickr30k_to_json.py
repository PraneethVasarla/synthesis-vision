from bs4 import BeautifulSoup
import json

with open("flickr30k.html","r") as file:
    html_data = file.read()

html = BeautifulSoup(html_data, features='html.parser')

image_data = {}
for link in html.find_all('a'):
    if link.get('href').endswith('.jpg'):
        image_name = link.text
        captions = [li.text for li in link.find_next('ul').find_all('li')]
        image_data[image_name] = captions

# Save as descriptions.json
with open('descriptions.json', 'w') as json_file:
    json.dump(image_data, json_file, indent=4)

print("Data saved as descriptions.json")