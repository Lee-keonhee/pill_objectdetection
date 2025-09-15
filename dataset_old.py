import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import json
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
import shutil
import random
import yaml

class CustomCocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        '''
        :param image_dir: image 폴더 경로
        :param annotation_file: 전체 annotation 파일(merged_annotation 경로)
        :param transforms: 이미지 변환
        '''
        self.image_dir = image_dir
        self.transforms= transforms
        try:
            with open (annotation_file,'r', encoding='utf-8') as f:
                self.coco = COCO()
                self.coco.dataset = json.load(f)
                self.coco.createIndex()
        except Exception as e:
            print(f"오류: 어노테이션 파일 로드 중 문제가 발생했습니다: {e}")
            # 오류가 발생하면 프로그램을 종료하거나 적절히 처리해야 합니다.
        # self.coco = COCO(annotation_file)                   # 전체 annotation 파일(merged_annotation) 불러오기
        self.ids = list(sorted(self.coco.imgs.keys()))      #

        self.original_id_to_sequential_label = {}  # 원본 ID -> 새로운 (1부터 시작하는) 순차적 레이블
        self.sequential_label_to_original_name = {}  # 새로운 순차적 레이블 -> 원본 이름

        sorted_original_category_ids = sorted(self.coco.cats.keys())

        sequential_label_counter = 1

        for original_cat_id in sorted_original_category_ids:
            original_cat_name = self.coco.cats[original_cat_id]['name']

            self.original_id_to_sequential_label[original_cat_id] = sequential_label_counter
            self.sequential_label_to_original_name[sequential_label_counter] = original_cat_name

            sequential_label_counter += 1

        # 총 클래스 개수는 실제 객체 클래스 수 (sequential_label_counter - 1) + 배경 클래스 (1)
        self.num_total_classes = sequential_label_counter

        # print(f"데이터셋에 총 {self.num_total_classes - 1}개의 실제 객체 클래스가 매핑되었습니다.")
        # print(f"새로운 레이블 매핑 (새 레이블 -> 원본 이름): {self.sequential_label_to_original_name}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_id = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_id)

        # 이미지 로드
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')

        temp_boxes = []
        temp_labels = []

        for i in range(len(coco_anns)):
            xmin = coco_anns[i]['bbox'][0]
            ymin = coco_anns[i]['bbox'][1]
            xmax = xmin + coco_anns[i]['bbox'][2]
            ymax = ymin + coco_anns[i]['bbox'][3]
            temp_boxes.append([xmin, ymin, xmax, ymax])

            # --- 원본 category_id를 새로운 순차적 레이블로 변환 ---
            original_category_id = coco_anns[i]['category_id']
            # 매핑된 레이블을 가져오거나, 매핑되지 않은 경우 0 (배경)으로 처리합니다.
            mapped_label = self.original_id_to_sequential_label.get(original_category_id, 0)
            temp_labels.append(mapped_label)

            # 실제 boxes와 labels에 할당
            # 만약 temp_boxes가 비어있다면, 모델 학습 시 오류를 방지하기 위해 더미 값을 추가합니다.
        boxes = []
        labels = []
        if not temp_boxes:
            # 예를 들어, 이미지 내에 객체가 없는 경우. Faster R-CNN 등은 빈 target을 싫어합니다.
            # 빈 이미지에 대한 더미 박스 (필요시): 이미지 크기에 맞는 더미 박스, 레이블은 배경(0)
            img_width, img_height = img.size  # PIL Image에서 크기 가져오기
            boxes.append([0.0, 0.0, img_width - 1.0, img_height - 1.0])  # 이미지 전체를 덮는 더미 박스
            labels.append(0)  # 배경 레이블 (0)
        else:
            boxes = temp_boxes
            labels = temp_labels

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        boxes_tensor = BoundingBoxes(boxes, format=BoundingBoxFormat.XYXY, canvas_size=img.size[::-1])

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels
        target["image_id"] = torch.tensor(img_id)   # image_id는 스칼라 텐서로 넣는 것이 일반적입니다.

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def custom_collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets          # 이미지와 타겟 딕셔너리들을 각각 리스트로 반환

''' 기존데이터 분할'''

# def merge_annotations(base_dir, output_path):
#     # COCO 형식의 기본 딕셔너리 구조
#     coco_format = {
#         "images": [],
#         "annotations": [],
#         "categories": [],
#         "info": {},
#         "licenses": []
#     }
#
#     # 카테고리 정보는 한 번만 추가 (여기서는 예시)
#     # 실제 데이터의 'categories' 정보를 파싱하여 추가해야 합니다.
#     category_id_map = {}
#
#     # 폴더 구조 순회
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith('.json'):
#                 json_path = os.path.join(root, file)
#                 # print(json_path)
#                 try:
#                     with open(json_path, 'r', encoding='utf-8') as f:
#                         data = json.load(f)
#                         # print(data)
#                     # images와 annotations 정보만 추출하여 추가
#                     if 'images' in data and data['images']:
#                         image_info = data['images'][0]
#                         coco_format["images"].append(image_info)
#
#                     if 'annotations' in data and data['annotations']:
#                         annotation_info = data['annotations'][0]
#                         coco_format["annotations"].append(annotation_info)
#
#                     # categories 정보 추가 (중복 방지)
#                     if 'categories' in data and data['categories']:
#                         category_info = data['categories'][0]
#                         category_id = category_info['id']
#                         if category_id not in category_id_map:
#                             coco_format["categories"].append(category_info)
#                             category_id_map[category_id] = True
#
#                 except json.JSONDecodeError as e:
#                     print(f"Skipping malformed JSON: {json_path}")
#
#     # 최종 JSON 파일 저장
#
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(coco_format, f, indent=4)
#
#     print(f"Merged JSON file saved at: {output_path}")
#
#
# def split_coco_dataset(original_coco_json, output_dir, val_split_ratio=0.2):
#     """
#     COCO 형식의 JSON 파일을 train/val 세트로 나눕니다.
#
#     Args:
#         original_coco_json (str): 원본 COCO JSON 파일 경로.
#         output_dir (str): 분할된 JSON 파일을 저장할 디렉터리.
#         val_split_ratio (float): validation 세트의 비율 (0.0 ~ 1.0).
#     """
#     with open(original_coco_json, 'r', encoding='utf-8') as f:
#         coco_data = json.load(f)
#
#     # 1. 전체 이미지 ID 목록 가져오기
#     all_image_ids = [img['id'] for img in coco_data['images']]
#     random.shuffle(all_image_ids)
#     print(all_image_ids)
#     print(set(all_image_ids))
#     print(len(all_image_ids))
#     print(len(set(all_image_ids)))
#     # 2. train/val 이미지 ID로 분할
#     num_val_images = int(len(all_image_ids) * val_split_ratio)
#     print(num_val_images)
#     val_image_ids = set(all_image_ids[:num_val_images])
#     print(len(val_image_ids))
#     train_image_ids = set(all_image_ids[num_val_images:])
#     print(len(train_image_ids))
#     # 3. 새로운 train/val 딕셔너리 구조 초기화
#     train_data = {
#         'images': [],
#         'annotations': [],
#         'categories': coco_data['categories']
#     }
#     val_data = {
#         'images': [],
#         'annotations': [],
#         'categories': coco_data['categories']
#     }
#
#     # 4. 이미지와 어노테이션 필터링
#     print("이미지 및 어노테이션 필터링 중...")
#     for img in coco_data['images']:
#         if img['id'] in train_image_ids:
#             train_data['images'].append(img)
#         elif img['id'] in val_image_ids:
#             val_data['images'].append(img)
#
#     for ann in coco_data['annotations']:
#         if ann['image_id'] in train_image_ids:
#             train_data['annotations'].append(ann)
#         elif ann['image_id'] in val_image_ids:
#             val_data['annotations'].append(ann)
#
#     # 5. 새로운 JSON 파일 저장
#     os.makedirs(output_dir, exist_ok=True)
#
#     train_json_path = os.path.join(output_dir, 'train_annotations.json')
#     val_json_path = os.path.join(output_dir, 'valid_annotations.json')
#
#     with open(train_json_path, 'w', encoding='utf-8') as f:
#         json.dump(train_data, f, indent=4)
#
#     with open(val_json_path, 'w', encoding='utf-8') as f:
#         json.dump(val_data, f, indent=4)
#
#     print(f"데이터셋 분할 완료!")
#     print(f"Train 세트: {len(train_data['images'])} 이미지, {len(train_data['annotations'])} 어노테이션")
#     print(f"Validation 세트: {len(val_data['images'])} 이미지, {len(val_data['annotations'])} 어노테이션")





###---------------------------------YOLO------------------------------------------###


#

def group_and_split_annotations(annotations_dir, output_dir, split_ratio=0.8):
    """
    모든 어노테이션을 이미지별로 그룹화하고, 훈련/검증 JSON 파일로 분할하여 저장합니다.
    """
    train_output_dir = os.path.join(output_dir, 'train')
    valid_output_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(valid_output_dir, exist_ok=True)

    print("모든 어노테이션을 이미지별로 그룹화 중...")
    image_annotations = {}
    annotation_count = 0

    # 모든 폴더 돌면서 json파일 찾음
    for root, _, files in os.walk(annotations_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # JSON 데이터에서 이미지명 추출
                        # "images" 리스트의 첫 번째 항목에 "file_name"이 있을 것으로 가정합니다.
                        if 'images' in data and data['images']:
                            image_name = data['images'][0]['file_name']

                            if image_name not in image_annotations:
                                image_annotations[image_name] = []

                            # 어노테이션 데이터를 딕셔너리에 추가
                            image_annotations[image_name].append(data)
                            annotation_count += 1

                except Exception as e:
                    print(f"오류 발생: {file_path} - {e}")

    print(f"총 {annotation_count}개의 어노테이션 파일을 {len(image_annotations)}개 이미지에 대해 그룹화했습니다.")

    image_names = list(image_annotations.keys())
    random.shuffle(image_names)

    split_index = int(len(image_names) * split_ratio)
    train_images = image_names[:split_index]
    valid_images = image_names[split_index:]

    print(f"\n훈련 이미지 수: {len(train_images)}개")
    print(f"검증 이미지 수: {len(valid_images)}개")

    train_annotations = []
    for img_name in train_images:
        train_annotations.extend(image_annotations[img_name])

    valid_annotations = []
    for img_name in valid_images:
        valid_annotations.extend(image_annotations[img_name])

    with open(os.path.join(train_output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)

    with open(os.path.join(valid_output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_annotations, f, indent=2, ensure_ascii=False)

    print(f"\n훈련 어노테이션: {len(train_annotations)}개 -> {os.path.join(train_output_dir, 'train.json')}으로 저장 완료.")
    print(f"검증 어노테이션: {len(valid_annotations)}개 -> {os.path.join(valid_output_dir, 'val.json')}으로 저장 완료.")


def copy_images_from_jsons(json_dir, source_images_dir, output_images_dir, image_name_key='file_name'):
    """
    JSON 파일을 기반으로 이미지를 훈련/검증 폴더로 복사합니다.
    """
    train_output_dir = os.path.join(output_images_dir, 'train')
    valid_output_dir = os.path.join(output_images_dir, 'val')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(valid_output_dir, exist_ok=True)

    def process_images(json_path, dest_dir):
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        print(f"\n{os.path.basename(json_path)} 파일에서 {len(annotations)}개의 어노테이션을 로드했습니다.")
        print(f"이미지를 {os.path.basename(dest_dir)} 폴더로 복사 중...")

        moved_count = 0
        copied_images = set()  # 중복 복사를 방지하기 위한 집합
        for annotation in annotations:
            # print(type(annotation))
            try:
                # 'images' 리스트를 순회하며 각 객체의 이미지 이름 추출
                # print(annotation.keys())
                for image_info in annotation.get('images', []):
                    # print(image_info)
                    image_name = image_info.get(image_name_key)
                    if image_name and image_name not in copied_images:
                        source_path = os.path.join(source_images_dir, image_name)
                        destination_path = os.path.join(dest_dir, image_name)

                        if os.path.exists(source_path):
                            shutil.copy(source_path, destination_path)
                            copied_images.add(image_name)
                            moved_count += 1
                        else:
                            print(f"경고: {source_path} 파일을 찾을 수 없습니다.")
            except KeyError:
                print(f"오류: JSON 파일에 '{image_name_key}' 키가 없습니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")

        print(f"{os.path.basename(dest_dir)} 데이터용 이미지 {moved_count}개 복사 완료.")

    process_images(os.path.join(json_dir, 'train/train.json'), train_output_dir)
    process_images(os.path.join(json_dir, 'val/val.json'), valid_output_dir)
    print("\n모든 이미지가 성공적으로 분할되었습니다.")




def coco2yolo(json_path, output_dir, train=True):
    if train:
        json_path = os.path.join(json_path, 'train', 'train.json')
        output_dir = os.path.join(output_dir, 'train')
    else:
        json_path = os.path.join(json_path, 'val', 'val.json')
        output_dir = os.path.join(output_dir, 'val')

    # 출력 디렉터리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 이미 처리한 파일들을 추적하기 위한 set
    processed_files = set()

    for item in data:
        images = item.get('images', [])
        annotations = item.get('annotations', [])

        if not images or not annotations:
            continue

        img_info = images[0]
        img_width = img_info['width']
        img_height = img_info['height']

        # YOLO 파일명 생성
        file_name = os.path.splitext(img_info['file_name'])[0] + '.txt'
        output_path = os.path.join(output_dir, file_name)

        # set에 파일이 없으면 'w' (덮어쓰기), 있으면 'a' (추가)
        if file_name not in processed_files:
            processed_files.add(file_name)
            write_mode = 'w'  # 첫 번째는 덮어쓰기
        else:
            write_mode = 'a'  # 이후는 추가

        yolo_annotations = []  # 한 이미지의 모든 어노테이션을 담을 리스트

        # 어노테이션 정보 순회
        for anno in annotations:
            category_id = anno['category_id']
            bbox = anno['bbox']

            # YOLO 좌표로 변환
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            yolo_width = bbox[2] / img_width
            yolo_height = bbox[3] / img_height

            # YOLO 포맷 문자열 생성
            yolo_string = f"{category_id} {x_center} {y_center} {yolo_width} {yolo_height}"
            yolo_annotations.append(yolo_string)

        # 모든 어노테이션 정보를 파일에 씁니다.
        with open(output_path, write_mode, encoding='utf-8') as f:
            if write_mode == 'a' and os.path.getsize(output_path) > 0:
                f.write('\n')
            f.write('\n'.join(yolo_annotations))

    print(f"변환 완료. YOLO 어노테이션 파일이 '{output_dir}'에 저장되었습니다.")



def create_yolo_yaml(path, train_dir, val_dir, class_names, output_path='dataset.yaml'):
    '''
    :param path: 데이터셋의 루트 경로입니다. (str)
    :param train_dir: 훈련 이미지 폴더의 상대 경로입니다. (str)
    :param val_dir: 검증 이미지 폴더의 상대 경로입니다. (str)
    :param class_names: 클래스 이름의 리스트입니다. (list)
    :param output_path: 생성될 YAML 파일의 저장 경로 및 파일명입니다. (str)
    :return:
        YOLO 학습에 필요한 YAML 데이터셋 구성 파일을 생성합니다.
    '''

    # 클래스 개수 계산
    nc = len(class_names)

    # YOLO 데이터셋 구성 딕셔너리 생성
    yolo_data = {
        'path': path,
        'train': train_dir,
        'val': val_dir,
        'nc': nc,
        'names': class_names
    }

    # YAML 파일로 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yolo_data, f, default_flow_style=False, allow_unicode=True)
        print(f"'{output_path}' 파일이 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"YAML 파일 생성 중 오류가 발생했습니다: {e}")


# PyYAML 라이브러리가 설치되어 있지 않다면 아래 명령어를 실행하세요.
# pip install PyYAML


def get_class_ids(json_path):
    """
    COCO JSON 파일에서 모든 고유한 클래스 ID를 추출합니다.

    Args:
        json_path (str): COCO JSON 파일의 경로.

    Returns:
        list: 모든 클래스 ID를 담은 리스트.
    """
    unique_ids = set()

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 데이터가 'images'나 'annotations'를 포함하는 단일 딕셔너리일 경우
        if isinstance(data, dict):
            categories = data.get('categories', [])
            for category in categories:
                unique_ids.add(category['id'])

        # 데이터가 각 이미지 정보가 담긴 딕셔너리 리스트일 경우
        elif isinstance(data, list):
            for item in data:
                categories = item.get('categories', [])
                for category in categories:
                    unique_ids.add(category['id'])

    except FileNotFoundError:
        print(f"오류: '{json_path}' 파일을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        print(f"오류: '{json_path}' 파일의 JSON 형식이 올바르지 않습니다.")
        return []

    # set을 list로 변환하여 반환
    return sorted(list(unique_ids))

if __name__ == "__main__":
    class Args:
        source_json_dir = './data/ai04-level1-project/train_annotations'
        output_json_dir = './data/labels'
        source_images_dir = './data/ai04-level1-project/train_images'
        output_images_dir = './data/images'
        dataset_root_path = './data'
        train_images_relative_path = 'images/train'
        valid_images_relative_path = 'images/val'
        output_file_name = 'my_dataset_config.yaml'

    args = Args
    # group_and_split_annotations(annotations_dir=args.source_json_dir,output_dir=args.output_json_dir, split_ratio=0.8)
    # copy_images_from_jsons(json_dir=args.output_json_dir, source_images_dir=args.source_images_dir, output_images_dir= args.output_images_dir, image_name_key= 'file_name')
    #
    # coco2yolo(json_path=args.output_json_dir, output_dir=args.output_json_dir)
    # coco2yolo(json_path=args.output_json_dir, output_dir=args.output_json_dir, train=False)
    # train_json_path = './data/labels/train/train.json'
    # class_ids = get_class_ids(train_json_path)
    #
    # create_yolo_yaml(
    #     path=args.dataset_root_path,
    #     train_dir=args.train_images_relative_path,
    #     val_dir=args.valid_images_relative_path,
    #     class_names=class_ids,
    #     output_path=args.output_file_name
    # )