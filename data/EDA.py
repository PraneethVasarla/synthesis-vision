import json

with open("flickr30k.json","r") as file:
    data = json.load(file)

print(len(data))