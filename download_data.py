"""

The data will be labeled according:
	0=Desert
	1=Vegetation
	2=Water
	3=Clouds
"""

import urllib
import os 

data_file = "data/test_images.txt"
label_file = "data/test_labels.txt"

output_dir = "data/TINY_SATELLITE"

if __name__ == '__main__':

	# Validate the file directories
	assert os.path.exists(data_file), "Cannot find the .txt file with images"
	assert os.path.exists(label_file), "Cannot find the .txt file with labels"

	if not os.path.exists(output_dir):
	    os.mkdir(output_dir)
	else:
	    print("Warning: output dir {} already exists".format(output_dir))

	# Read in and save the file URL one per line
	image_url_list = []
	label_list = []
	with open(data_file) as f:
	    image_url_list = f.read().splitlines()
	with open(label_file) as f:
	    label_list = f.read().splitlines()

	# Check the number of labels matches number of images
	assert len(label_list) == len(image_url_list)
	print("Read in %s files" % len(image_url_list))

	# Save each file with its corresponding label
	for idx, (label, file_url) in enumerate(zip(label_list, image_url_list)):
	    urllib.urlretrieve(file_url, "%s/%s-%s.jpg" % (output_dir, label, str(idx).zfill(2)))

	print("Done downloading dataset")
