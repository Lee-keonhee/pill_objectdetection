import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import json
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


class CustomCocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        '''
        :param image_dir: image 폴더 경로
        :param annotation_file: 전체 annotation 파일(merged_annotation 경로)
        :param transforms: 이미지 변환
        '''
        self.image_dir = image_dir
        self.transforms= transforms
        self.coco = COCO(annotation_file)                   # 전체 annotation 파일(merged_annotation) 불러오기
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


def merge_annotations(base_dir, output_path):
    # COCO 형식의 기본 딕셔너리 구조
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {},
        "licenses": []
    }

    # 카테고리 정보는 한 번만 추가 (여기서는 예시)
    # 실제 데이터의 'categories' 정보를 파싱하여 추가해야 합니다.
    category_id_map = {}

    # 폴더 구조 순회
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                # print(json_path)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # print(data)
                    # images와 annotations 정보만 추출하여 추가
                    if 'images' in data and data['images']:
                        image_info = data['images'][0]
                        coco_format["images"].append(image_info)

                    if 'annotations' in data and data['annotations']:
                        annotation_info = data['annotations'][0]
                        coco_format["annotations"].append(annotation_info)

                    # categories 정보 추가 (중복 방지)
                    if 'categories' in data and data['categories']:
                        category_info = data['categories'][0]
                        category_id = category_info['id']
                        if category_id not in category_id_map:
                            coco_format["categories"].append(category_info)
                            category_id_map[category_id] = True

                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON: {json_path}")

    # 최종 JSON 파일 저장

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=4)

    print(f"Merged JSON file saved at: {output_path}")
