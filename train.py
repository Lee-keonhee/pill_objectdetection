import torch
import matplotlib.pyplot as plt
import ssd_trainer
import yolo_trainer

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우
plt.rcParams['axes.unicode_minus'] = False

class Args:
    def __init__(self):
        # Data paths - 새로운 구조에 맞게 수정
        self.train_image_dir = "./data/images/train"  # 훈련 이미지 폴더
        self.valid_image_dir = "data/images/val"  # 검증 이미지 폴더
        self.train_annotation_path = "./data/labels/train/train.json"  # 훈련 어노테이션 파일
        self.valid_annotation_path = "data/labels/val/valid.json"  # 검증 어노테이션 파일
        self.checkpoint_dir = "./checkpoints"  # checkpoint 모델 경로
        self.yaml_path = './my_dataset_config.yaml'

        # Training parameters
        self.batch_size = 4  # 배치 크기
        self.num_epochs = 11  # 에포크 수
        self.learning_rate = 0.005  # 학습률
        self.weight_decay = 0.0005  # 학습률 변화
        self.momentum = 0.9  # 모멘텀
        self.step_size = 3
        self.gamma = 0.1

        # Model parameters
        self.num_classes = None  # dataset에서 자동으로 결정, 손수 결정할 때만 입력
        self.model_name = 'CustomFasterRCNN'  # ['Yolov#', 'CustomFasterRCNN', ]

        # Training settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 디바이스 설정
        self.num_workers = 4  # Dataloader의 num_worker 수 설정
        self.print_freq = 10  # 얼마마다 loss 프린트 할 것인지
        self.save_freq = 1  # 체크포인트를 몇 epoch마다 저장할 것인지

        # Resume training
        # 중간에 학습 멈추고 다시 시작할때, checkpoint 불러오는 설정
        self.resume = False  # False는 안불러옴 - 처음부터 학습함(대신 이전에 학습해서 checkpoint있으면, 덮어씀.
        self.checkpoint_path = None
        # self.resume = True                                                      # True는 불러옴
        # self.checkpoint_path = "./checkpoints/checkpoint_epoch_10.pth"          # 불러올 checkpoint 경로

        # Validation - 이제 별도 파일로 제공되므로 필요없음
        # self.val_split = 0.2

        # Visualization
        self.visualize_predictions = True  # validation 이미지 보여 주는가
        self.vis_num_samples = 5  # 이미지 갯수

        # WandB settings
        self.use_wandb = True  # wandb 연결
        self.wandb_project = "object-detection"
        self.wandb_entity = 'AI-team4'
        self.wandb_run_name = None  # None으로 설정되면 아무렇게나 자동으로 저장됨

        # Evaluation settings
        self.eval_freq = 1  # 몇 epoch마다 평가하나
        self.confidence_threshold = 0.5  # classification된 박스를 제거하는 기준
        self.nms_threshold = 0.5  # IoU를 기준으로 박스를 제거하는 기준                                         # IoU를 기준으로 박스를 제거하는 기준


def main():
    args = Args()
    model_name = input('사용하실 모델 명을 입력하세요. ["FasterRCNN","Yolo"] :')
    model_name = str.lower(model_name)
    print(args)
    if model_name == 'fasterrcnn':
        args.model_name = 'CustomFasterRCNN'
        ssd_trainer.main(args)

    elif model_name == 'yolo':
        args.model_name = 'yolov8n.pt'
        yolo_trainer.main(args)
    # print(f'사용중인 모델 명 : {args.model_name}')

if __name__ == "__main__":
    main()
