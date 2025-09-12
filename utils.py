import os
import json
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

def bring_annotation(image_base_dir, annotation_base_dir, image_filename: str):
    '''
    :param image_base_dir: 이미지 데이터 경로 (ex. './data/ai04-level1-project/train_images')
    :param annotation_base_dir: 어노테이션 base 폴더 경로 (ex. './data/ai04-level1-project/train_annotations')
    :param image_list: 이미지의 파일명( K-001900-010224-016551-031705_0_2_0_2_70_000_200.png )을 가지는 리스트
    :return: bbox_list, label_list, id_to_name
    파일 경로를 통해, 각 annotation에서의 bbox, label, id_to_name 반환 (이미지당 boundbox가 3개면 길이가 3의 bbox_list, label_list, id_to_name 딕셔너리 반환)
    '''
    image_filename = os.path.splitext(image_filename)[0]
    splited_filename = image_filename.split('_')
    annotation_dir = os.path.join(annotation_base_dir, splited_filename[0] + '_json')
    pill_dir = os.listdir(annotation_dir)
    pill_annotation_base_dir = [os.path.join(annotation_dir, pill_name) for pill_name in pill_dir]
    pill_annotation_path = [os.path.join(base_dir, image_filename) for base_dir in pill_annotation_base_dir]
    pill_annotation_path = [path_item + '.json' for path_item in pill_annotation_path]
    # anno_list= []
    bbox_list = []
    label_list = []
    id_to_name = {}
    for anno_path in pill_annotation_path:
        try:
            with open(anno_path, 'r', encoding='utf-8') as f:
                anno = json.load(f)
            image_info = anno['images'][0]
            annotation_info = anno['annotations'][0]
            category_info = anno['categories'][0] if anno['categories'] else {'name': 'Unknown'}
            # anno_list.append(anno)
            bbox_list.append(annotation_info['bbox'])
            label_list.append(annotation_info['category_id'])
            if annotation_info['category_id'] not in id_to_name.keys():
                id_to_name[annotation_info['category_id']] = category_info['name']

        except:
            pass
    return bbox_list, label_list, id_to_name


def check_annotation_file(bbox_list: list):
    '''
    bring_annotation의 반환값 (bbox_list:list, label_list:list, id_to_name:dict)
    을 활용하여, annotation파일 손상 및 결측 데이터 or inf 데이터 확인
    '''
    return

def image_check(image_base_dir, annotation_base_dir, image_list, num_image=10, random_seed=42):
    '''
    :param image_base_dir: 이미지 데이터 경로 (ex. './data/ai04-level1-project/train_images')
    :param annotation_base_dir: 어노테이션 base 폴더 경로 (ex. './data/ai04-level1-project/train_annotations')
    :param image_list: 이미지의 파일명( K-001900-010224-016551-031705_0_2_0_2_70_000_200.png )을 가지는 리스트
    :return: 없음 id2name_dict: category_id를 이름으로 변환해주는 dictionary
    이외 이미지를 plot해줌
    '''
    unique_label = set()
    id_to_name_dict = {}
    np.random.seed(random_seed)
    selected_num_list = np.random.randint(0,len(image_list),num_image)
    selected_image_list = [image_list[i] for i in selected_num_list]
    for idx, image_filename in enumerate(selected_image_list):
        bbox_list, label_list , id_to_name = bring_annotation(image_base_dir, annotation_base_dir, image_filename)

        ### 이미지 시각화
        img = cv2.imread(os.path.join(image_base_dir, image_filename))
        if img is None:
            print(f'이미지 파일 로드 실패 : {image_filename}')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for bbox, label in zip(bbox_list, label_list):
            # print(label)
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            if all(v is not None for v in [xmin, ymin, xmax, ymax]):
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(img, 'K-' + str(label), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Image: {image_filename}")
        plt.axis('off')
        plt.show()


def get_id2name_dict(image_base_dir, annotation_base_dir, image_list):
    '''
    :param image_base_dir: 이미지 데이터 경로 (ex. './data/ai04-level1-project/train_images')
    :param annotation_base_dir: 어노테이션 base 폴더 경로 (ex. './data/ai04-level1-project/train_annotations')
    :param image_list: 이미지의 파일명( K-001900-010224-016551-031705_0_2_0_2_70_000_200.png )을 가지는 리스트
    :return: id2name_dict: category_id를 이름으로 변환해주는 dictionary
    '''
    unique_label = set()
    id2name_dict = {}
    for idx, image_filename in enumerate(image_list):
        bbox_list, label_list, id2name = bring_annotation(image_base_dir, annotation_base_dir, image_filename)

        unique_label.update(label_list)
        id2name_dict.update(id2name)

    return id2name_dict



def visualize_prediction(image, prediction, classes, target=None):
# def visualize_prediction(image, prediction):
    """
    image (torch.Tensor): 추론에 사용된 이미지 (C, H, W 형식).
    prediction (dict): 모델의 예측 결과 (boxes, labels, scores 포함).
    classes (list): 클래스 이름 리스트.
    """
    # Tensor 이미지를 (H, W, C) 형식으로 변환
    image = image.permute(1, 2, 0).numpy()

    # Matplotlib을 사용한 이미지 시각화
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Bounding Box와 클래스 이름 시각화
    ## prediction의 bounging box 시각화
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score > 0.5:  # Confidence Score 임계값
            x_min, y_min, x_max, y_max = box.tolist()
            width, height = x_max - x_min, y_max - y_min
            # Bounding Box 추가
            rect = patches.Rectangle(
                (x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

            # 클래스 이름과 Confidence Score 추가
            ax.text(
                x_min,
                y_min - 10,
                f"{classes[label.item()]}: {score:.2f}",
                # f"{label}: {score:.2f}",
                color="red",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )

    ## target이 있을 경우, target의 bounding box 시각화
    if target:
        for box, label in zip(target['boxes'], target['labels']):
            x_min, y_min, x_max, y_max = box.tolist()
            width, height = x_max - x_min, y_max - y_min
            # Bounding Box 추가
            rect = patches.Rectangle(
                (x_min, y_min), width, height, linewidth=2, edgecolor="blue", facecolor="none"
            )
            ax.add_patch(rect)

            ax.text(
                x_min,
                y_max + 10,
                f"{classes[label.item()]}",
                # f"{label}: {score:.2f}",
                color="blue",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )
        plt.axis("off")
        plt.show()