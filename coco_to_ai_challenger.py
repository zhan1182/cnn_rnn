
import sys
import json

def main():

	coco_json = sys.argv[1]
	ai_json = sys.argv[2]

	coco = json.load(open(coco_json))

	coco_id_to_filename_mapping = {}
	filename_to_caption_mapping = {}

	coco_images = coco['images']
	for d in coco_images:
		coco_id_to_filename_mapping[d['id']] = d['file_name']

	annotations = coco['annotations']		
	for anna in annotations:
		fn = coco_id_to_filename_mapping[anna['image_id']]

		if fn not in filename_to_caption_mapping:
			filename_to_caption_mapping[fn] = [anna['caption']]
		else:
			filename_to_caption_mapping[fn].append(anna['caption'])


	filename_to_caption_mapping_list = []
	for fn, captions in filename_to_caption_mapping.items():
		d = {'image_id': fn, 'caption': captions}
		filename_to_caption_mapping_list.append(d)

	json.dump(filename_to_caption_mapping_list, open(ai_json, 'wb'))


if __name__ == '__main__':
	main()