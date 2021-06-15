import requests
import json
import os

query_items = [
	"elephants",
	"lion",
	"bear",
	"giraffe",
	"antelope",
	"hippo",
	"hare",
	"zebra",
	"rhino",
	"baboon",
	"cheetah",
	"deer",
	"tiger",
	"leopard",
	"bird"
]

proxy = {
    "https": 'https://203.243.63.16:80',
    "http": 'https://203.243.63.16:80' 
}

for query in query_items:
	# query = query.replace(" ","-")
	print(query)
	if not os.path.exists(os.path.join("links",query)):
		os.makedirs(os.path.join("links",query))
	else:
		continue
	page_number = 1
	while True:
		url1 = "https://unsplash.com/napi/search/photos?query={}&per_page=20&page={}&xp=".format(query, page_number)
		r = requests.get(url1)
		data = dict(r.json())
		if not data["results"]:
			break
		with open("./links/{}/{}-links-{}.json".format(query, query, page_number), "w") as outfile: 
			json.dump(data, outfile,indent = 3)
		page_number += 1
