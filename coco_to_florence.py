import os
import json
import random

# 1. download the dataset from https://huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers
# 2. move annotations_with_caption.json to the ./annotations folder
# 3. unzip the arxiv_paper_images.zip and move the images to the ./images folder
# 4. run the script: python coco_to_florence.py

train_percentage = 0.85
coco_json_dir = "./annotations/annotations_with_caption.json" # we take coco format dataset by default
# coco_json_dir = "./annotations/annotations_no_caption.json" # we take coco format dataset by default
output_dir = "./annotations"

def convert_to_florence_format(coco_json_dir, output_dir):

	# the code here is very messy because it was easy to debug and understand

	print("start converting coco annotations to florence format...")

	with open(coco_json_dir, 'r') as file:
		data = json.load(file)

	category_dict = {category['id']: category['name'] for category in data['categories']}
	print("labels :", category_dict)
	
	img_dict = {}
	for img in data['images']:
		img_dict[img['id']] = {
			'width': img['width'],
			'height': img['height'],
			'file_name': img['file_name'],
			'annotations': [],
			'annotations_str': ""
		}

	annotation_dict = {annotation['image_id']: annotation['bbox'] for annotation in data['annotations']}

	def format_annotation(annotation):
		category_id = annotation['category_id']
		bbox = annotation['bbox'] # coco bbox format: [x, y, width, height]
		this_image_width = img_dict[int(annotation['image_id'])]['width']
		this_image_height = img_dict[int(annotation['image_id'])]['height']
		# normalize the numbers to be between 0 and 1 then multiplied by 1000.
		# forence 2 format: label<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>
		x1 = int(bbox[0] / this_image_width * 1000)
		y1 = int(bbox[1] / this_image_height * 1000)
		x2 = int((bbox[0] + bbox[2]) / this_image_width * 1000)
		y2 = int((bbox[1] + bbox[3]) / this_image_height * 1000)

		return f"{category_dict[category_id]}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"

	for annotation in data['annotations']:
		try:
			annotation_str = format_annotation(annotation)
			if annotation['image_id'] in img_dict:
				img_dict[annotation['image_id']]['annotations'].append(annotation_str)
		except:
			continue

	florence_data = []
	for img_id, img_data in img_dict.items():
		annotations_str = "".join(img_data['annotations'])

		if len(annotations_str) > 0:
			florence_data.append({
				"image": img_data['file_name'],
				"prefix": "<OD>",
				"suffix": annotations_str
			})
		else:
			# OPTIONAL: some images have no annotations, you can choose to ignore them
			# only randomly sample 5% of the images without annotations
			if random.random() < 0.05:
				florence_data.append({
					"image": img_data['file_name'],
					"prefix": "<OD>",
					"suffix": ""
				})

	print("total number of images:", len(florence_data))

	# split the data into train and test and save them into jsonl files
	train_split = int(len(florence_data) * train_percentage)
	train_data = florence_data[:train_split]
	test_data = florence_data[train_split:]

	print("train size:", len(train_data))
	print("test size:", len(test_data))

	train_output_dir = os.path.join(output_dir, "train.jsonl")
	test_output_dir = os.path.join(output_dir, "test.jsonl")

	# save train and test data into jsonl files
	if os.path.exists(train_output_dir):
		os.remove(train_output_dir)

	with open(train_output_dir, 'w') as file:
		for entry in train_data:
			json.dump(entry, file)
			file.write("\n")
	
	if os.path.exists(test_output_dir):
		os.remove(test_output_dir)

	with open(test_output_dir, 'w') as file:
		for entry in test_data:
			json.dump(entry, file)
			file.write("\n")
	
	print("train and test data saved to ", output_dir)
	print("Now you can run \"accelerate launch train.py\" to train the model.")

convert_to_florence_format(coco_json_dir, output_dir)