import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.transforms import v2

import models
from dataset_old import CustomCocoDataset, custom_collate_fn
from models import get_model, CustomFasterRCNN
from utils import visualize_prediction, get_id2name_dict

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우
plt.rcParams['axes.unicode_minus'] = False

from ultralytics import YOLO

def main(args):
    # model = YOLO(args.model_name)
    # results = model.train(data = args.yaml_path,
    #                       epochs=args.num_epochs,
    #                       imgsz=640,
    #                       batch=args.batch_size,
    #                       device=0)
    #                       # device=args.device)
    # print("학습 완료! best.pt 저장 위치:", model.ckpt_path)

    # 학습된 모델 불러오기
    trained_model = YOLO("./runs/detect/train/weights/best.pt")

    # 추론 실행
    results = trained_model.predict(
        source="./data/ai04-level1-project/test_images",  # 단일 이미지, 폴더, 비디오 모두 가능
        save=True,  # 결과 이미지 저장
        conf=0.5  # confidence threshold
    )

    # 결과 확인
    for r in results:
        print(r.boxes.xyxy)  # 감지된 박스 좌표
        print(r.boxes.conf)  # confidence 값
        print(r.boxes.cls)  # 클래스 ID

if __name__ == "__main__":
    args = Args()
    main(args)
