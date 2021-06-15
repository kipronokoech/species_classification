import os

print(os.listdir("./images"))

for species in os.listdir("./images"):
	files = os.listdir(os.path.join("images/",species))
	for file in files:
		file1 = os.path.join("images",species,file)
		print(file1)
		filename, ext = os.path.splitext(file1)
		filename_split = filename.split("_")
		filename_split[-1] = species
		new_filename = "_".join(filename_split)+ext
		os.rename(file1, new_filename)
		print(new_filename)
		print(20*"-")

		# print(os.path.splitext(file1))
		# print(filename_split)