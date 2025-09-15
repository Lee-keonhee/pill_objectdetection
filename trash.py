from utils import get_id2name_dict, create_class_mapping
import os
image_base_dir= './data/images/train'
annotation_base_dir = './data/labels/train'
annotation_train_path = './data/labels/train/train.json'
# a = os.listdir(image_base_dir)
# print(a)
# get_id2name_dict(image_base_dir=image_base_dir, annotation_base_dir=annotation_base_dir, image_list=a)

dict1, dict2, dict3 = create_class_mapping(input_json_path=annotation_train_path,output_json_path= './label2idname.json')

print(dict1)
print(dict2)
print(dict3)
