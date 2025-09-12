import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights  # weights 가져오기

def get_model(model_name, num_classes):
    """모델 이름에 따라 해당하는 모델 클래스의 인스턴스를 반환합니다. [사용가능한 모델명 : 'CustomFasterRCNN', '']
    """
    model_zoo = {
        "CustomFasterRCNN": CustomFasterRCNN,
        # 여기에 새로운 모델 클래스를 추가하면 됩니다.
        # "CustomYOLOv8": CustomYOLOv8,
    }

    if model_name not in model_zoo:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")

    model_class = model_zoo[model_name]
    return model_class(num_classes)


class CustomFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CustomFasterRCNN, self).__init__()

        # 1. ImageNet COCO 데이터셋으로 사전 학습된 Faster R-CNN 모델 불러오기
        # 'DEFAULT'는 최신 권장 가중치를 의미합니다.
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # 2. 분류기(classifier)의 입력 특성 수 확인
        # Faster R-CNN의 box_predictor는 cls_score(클래스 점수)와 bbox_pred(바운딩 박스 예측)를 포함합니다.
        # cls_score의 in_features를 가져와야 합니다.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # 3. 새로운 헤드(FastRCNNPredictor)로 교체
        # 기존 클래스 수 대신 우리 데이터셋의 num_classes를 사용합니다.
        # 주의: 객체 탐지 모델의 num_classes는 '클래스 개수 + 1 (배경 클래스)' 입니다.
        # 따라서 외부에서 전달받는 num_classes는 이미 배경 클래스를 포함한 개수여야 합니다.
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        # Faster R-CNN 모델의 forward는 학습 시에는 images와 targets를 모두 받습니다.
        # 평가 시에는 images만 받습니다.
        return self.model(images, targets)

