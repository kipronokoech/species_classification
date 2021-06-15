import os
import shutil


a = [len(os.listdir(os.path.join("images",i))) for i in os.listdir("images") if i!="exclude.txt"]

print(sum(a))

b = [len(os.listdir(os.path.join("annotations",i,"images"))) for i in os.listdir("annotations") if i!="via.html"]

print(sum(b))








exit()
done1= os.listdir("annotations/done_patch/images")
for index1, folder in enumerate(os.listdir("images")):
	files = os.listdir(os.path.join("images",folder))
	for index, file in enumerate(files):
		current_path = os.path.join("images",folder, file)
		if file in done1:
			os.remove(current_path)


for i in range(30):
	for index1, folder in enumerate(os.listdir("images")):

		files = os.listdir(os.path.join("images",folder))
		num_files = [len(os.listdir(os.path.join("images",folder1))) for folder1 in os.listdir("images")]
		print(max(num_files), num_files)

		if max(num_files)>=0:
			files = os.listdir(os.path.join("images",folder))[:50]
		else:
			break

		if not os.path.isdir("annotations/patch{}/images".format(i)):
			os.makedirs("annotations/patch{}/images".format(i))
		elif not os.path.isdir("annotations/patch{}/annotation_files".format(i)):
			os.makedirs("annotations/patch{}/annotation_files".format(i))
			
		for index, file in enumerate(files):
			current_path = os.path.join("images",folder, file)
			new_path = os.path.join("annotations/patch{}/images".format(i), file)
			shutil.move(current_path, new_path)
	break

	