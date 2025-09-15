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

from utils import create_class_mapping
def create_coco_format(data_list):
    """
    딕셔너리 리스트를 하나의 COCO 포맷 딕셔너리로 병합합니다.
    """
    merged_data = {
        'images': [],
        'annotations': [],
        'categories': [],
        'info': {},
        'licenses': []
    }

    unique_categories = {}

    for item in data_list:
        merged_data['images'].extend(item.get('images', []))
        merged_data['annotations'].extend(item.get('annotations', []))

        for cat in item.get('categories', []):
            if cat['id'] not in unique_categories:
                unique_categories[cat['id']] = cat

    merged_data['categories'] = list(unique_categories.values())

    return merged_data


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

    for root, _, files in os.walk(annotations_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        if 'images' in data and data['images']:
                            image_name = data['images'][0]['file_name']
                            if image_name not in image_annotations:
                                image_annotations[image_name] = []
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

    # --- 이 부분이 핵심입니다. ---
    # 분할된 어노테이션 리스트를 단일 COCO 포맷으로 변환합니다.
    train_data = create_coco_format(train_annotations)
    valid_data = create_coco_format(valid_annotations)

    with open(os.path.join(train_output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(valid_output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

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

        print(f"\n{os.path.basename(json_path)} 파일에서 {len(annotations['images'])}개의 어노테이션을 로드했습니다.")
        print(f"이미지를 {os.path.basename(dest_dir)} 폴더로 복사 중...")

        moved_count = 0
        copied_images = set()  # 중복 복사를 방지하기 위한 집합
        for annotation in annotations['images']:
            # print(type(annotation))
            try:
                # 'images' 리스트를 순회하며 각 객체의 이미지 이름 추출
                # print(annotation.keys())
                # for image_info in annotation.get('images', []):
                    # print(image_info)
                # image_name = image_info.get(image_name_key)
                image_name = annotation.get(image_name_key)
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




def coco2yolo(json_path, output_dir, label_change_dict, train=True):
    if train:
        json_path = os.path.join(json_path, 'train', 'train.json')
        output_dir = os.path.join(output_dir, 'train')
    else:
        json_path = os.path.join(json_path, 'val', 'valid.json')
        output_dir = os.path.join(output_dir, 'val')


    # 출력 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)

    # COCO 객체 로드
    # coco = COCO(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = COCO()
        coco.dataset = json.load(f)
        coco.createIndex()
    # 이미지 ID와 파일명 매핑
    image_ids = coco.getImgIds()

    print(f"총 {len(image_ids)}개의 이미지를 변환합니다.")

    # 각 이미지를 순회하며 YOLO 파일 생성
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_width = img_info['width']
        img_height = img_info['height']
        file_name = os.path.splitext(img_info['file_name'])[0] + '.txt'
        output_path = os.path.join(output_dir, file_name)

        # 이미지에 해당하는 모든 어노테이션 정보 가져오기
        annotation_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(annotation_ids)

        yolo_lines = []
        for anno in annotations:
            category_id = anno['category_id']
            # print(category_id)
            label = label_change_dict[str(category_id)]
            bbox = anno['bbox']
            # YOLO 좌표로 변환 (정규화된 중심점과 너비/높이)
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            yolo_width = bbox[2] / img_width
            yolo_height = bbox[3] / img_height

            # YOLO 포맷 문자열 생성
            yolo_string = f"{label} {x_center} {y_center} {yolo_width} {yolo_height}"
            yolo_lines.append(yolo_string)

        # 한 이미지의 모든 어노테이션 정보를 한 번에 파일에 씁니다.
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

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

class Args:
    source_json_dir = './data/ai04-level1-project/train_annotations'
    output_json_dir = './data/labels'
    source_images_dir = './data/ai04-level1-project/train_images'
    output_images_dir = './data/images'
    dataset_root_path = './data'
    train_images_relative_path = 'images/train'
    valid_images_relative_path = 'images/val'
    output_file_name = 'my_dataset_config.yaml'
    class_mapping_json_path = './data/labels/train/train.json'

if __name__ == "__main__":


    # args = Args()


    # group_and_split_annotations(annotations_dir=args.source_json_dir,output_dir=args.output_json_dir, split_ratio=0.8)
    # copy_images_from_jsons(json_dir=args.output_json_dir, source_images_dir=args.source_images_dir, output_images_dir= args.output_images_dir, image_name_key= 'file_name')


    # id2label, label2id, label2name = create_class_mapping(input_json_path=args.class_mapping_json_path)
    # coco2yolo(json_path=args.output_json_dir, output_dir=args.output_json_dir, label_change_dict=id2label)
    # coco2yolo(json_path=args.output_json_dir, output_dir=args.output_json_dir, label_change_dict=id2label, train=False)


    # train_json_path = './data/labels/train/train.json'
    # class_ids = get_class_ids(train_json_path)
    # names_list = list(label2name.values())
    # create_yolo_yaml(
    #     path=args.dataset_root_path,
    #     train_dir=args.train_images_relative_path,
    #     val_dir=args.valid_images_relative_path,
    #     class_names=names_list,
    #     output_path=args.output_file_name
    # )