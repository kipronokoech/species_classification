from urllib.request import urlretrieve
import os
import json
from urllib.error import HTTPError, URLError
import time
# folders = os.listdir("links")
folders = [
	"elephants",
	"lion",
	"bears",
	"giraffe",
	"antelope",
	"hippo",
	"hare",
	"zebra"
]


for folder in folders:
	if not os.path.exists(os.path.join("images",folder)):
		os.makedirs(os.path.join("images",folder))
	if len(os.listdir(os.path.join("images",folder))) != 0:
		continue
	files = os.listdir(os.path.join("links",folder))
	index = 0
	for i in range(1,len(files)):
		if len(os.listdir(os.path.join("images",folder)))>1000:
			break
		file = "{}-links-{}.json".format(folder, i)
		file_path = os.path.join("links", folder, file)
		datas = json.load(open(file_path))["results"]
		done = [i.split("_")[-2] for i in os.listdir(os.path.join("images",folder))]
		print(file)
		for data in datas:
			id1 = data["id"]
			if id1 in done:
				index += 1
				continue
			if index >= 1000:
				break
			url = data["urls"]["full"]
			name = "IMG_{}_{}_{}.jpg".format(index, id1, folder)
			to_path = os.path.join("images",folder,name)
			try:
				urlretrieve(url, to_path)
			except (HTTPError, URLError) as e:
				print("errors")
				time.sleep(30)
				continue
			index += 1

#urlretrieve(Miniature_Image_URL, Miniature_Image_Name)