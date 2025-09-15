from dataset_old import coco2yolo
import json



train_data_path = './train_annotations.json'
valid_data_path = './val_annotations.json'
coco2yolo(train_data_path,output_dir='./yolo_data/train')

coco2yolo(valid_data_path, output_dir='./yolo_data/valid')