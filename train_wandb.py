import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.transforms import v2
import numpy as np
import wandb
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile

import models
from dataset_old import CustomCocoDataset, custom_collate_fn, merge_annotations
from models import get_model, CustomFasterRCNN
from utils import visualize_prediction, get_id2name_dict

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우
plt.rcParams['axes.unicode_minus'] = False


class Args:
    def __init__(self):
        # Data paths
        self.image_dir = "./data/ai04-level1-project/train_images"
        self.annotation_dir = "./data/ai04-level1-project/train_annotations"
        self.merged_annotation_path = "./merged_annotations.json"
        self.checkpoint_dir = "./checkpoints"

        # Training parameters
        self.batch_size = 4
        self.num_epochs = 12
        self.learning_rate = 0.005
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.step_size = 3
        self.gamma = 0.1

        # Model parameters
        self.num_classes = None  # Will be set automatically from dataset
        self.model_name = 'CustomFasterRCNN'

        # Training settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        self.print_freq = 10
        self.save_freq = 1  # Save checkpoint every N epochs

        # Resume training
        # self.resume = False
        # self.checkpoint_path = None
        self.resume = True
        self.checkpoint_path = "./checkpoints/checkpoint_epoch_10.pth"

        # Validation
        self.val_split = 0.2  # 20% for validation
        self.label2id = './label2id.json'

        # Visualization
        self.visualize_predictions = True
        self.vis_num_samples = 5

        # WandB settings
        self.use_wandb = True
        self.wandb_project = "Object-Detection"
        self.wandb_entity = 'AI-team4'  # Your wandb username/team
        self.wandb_run_name = None  # Will be auto-generated if None

        # Evaluation settings
        self.eval_freq = 1  # Evaluate every N epochs
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.5


def get_transforms(train=True):
    """데이터 증강을 위한 transform 정의"""
    if train:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPhotometricDistort(p=0.5),
        ])
    else:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    return transforms


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, use_wandb=False):
    """한 에폭 학습 함수"""
    model.train()
    running_loss = 0.0
    running_loss_components = {}

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Loss components tracking
        for key, value in loss_dict.items():
            if key not in running_loss_components:
                running_loss_components[key] = 0.0
            running_loss_components[key] += value.item()

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if batch_idx % print_freq == 0:
            print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")

            # Log batch-level metrics to WandB
            if use_wandb:
                wandb.log({
                    "batch_loss": losses.item(),
                    "batch": epoch * len(data_loader) + batch_idx
                })

    # Calculate average losses
    avg_loss = running_loss / len(data_loader)
    avg_loss_components = {k: v / len(data_loader) for k, v in running_loss_components.items()}

    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

    return avg_loss, avg_loss_components


def evaluate_with_coco_metrics(model, data_loader, device, coco_gt, label2catid, confidence_threshold=0.5):
    """COCO 메트릭을 사용한 평가"""
    model.eval()
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images_gpu = [image.to(device) for image in images]
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Loss calculation (set model to train mode temporarily)
            model.train()
            loss_dict = model(images_gpu, targets_gpu)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # Predictions (set model back to eval mode)
            model.eval()
            outputs = model(images_gpu)

            # Process predictions for COCO evaluation
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                # Filter by confidence threshold
                keep_indices = scores > confidence_threshold
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]

                # Convert to COCO format
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j]
                    width = x2 - x1
                    height = y2 - y1

                    predictions.append({
                        'image_id': image_id,
                        'category_id': label2catid[int(labels[j])],
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'score': float(scores[j])
                    })

    avg_loss = total_loss / len(data_loader)

    # COCO evaluation
    if len(predictions) > 0:
        # Create temporary file for predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(predictions, tmp_file)
            pred_file = tmp_file.name

        try:
            # Load predictions
            coco_pred = coco_gt.loadRes(pred_file)

            # Evaluate
            coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Extract metrics
            metrics = {
                'mAP_0.5:0.95': coco_eval.stats[0],
                'mAP_0.5': coco_eval.stats[1],
                'mAP_0.75': coco_eval.stats[2],
                'mAP_small': coco_eval.stats[3],
                'mAP_medium': coco_eval.stats[4],
                'mAP_large': coco_eval.stats[5],
                'mAR_1': coco_eval.stats[6],
                'mAR_10': coco_eval.stats[7],
                'mAR_100': coco_eval.stats[8],
                'mAR_small': coco_eval.stats[9],
                'mAR_medium': coco_eval.stats[10],
                'mAR_large': coco_eval.stats[11],
            }

        except Exception as e:
            print(f"Error in COCO evaluation: {e}")
            metrics = {key: 0.0 for key in [
                'mAP_0.5:0.95', 'mAP_0.5', 'mAP_0.75', 'mAP_small', 'mAP_medium', 'mAP_large',
                'mAR_1', 'mAR_10', 'mAR_100', 'mAR_small', 'mAR_medium', 'mAR_large'
            ]}

        finally:
            # Clean up temporary file
            if os.path.exists(pred_file):
                os.unlink(pred_file)
    else:
        print("No predictions made!")
        metrics = {key: 0.0 for key in [
            'mAP_0.5:0.95', 'mAP_0.5', 'mAP_0.75', 'mAP_small', 'mAP_medium', 'mAP_large',
            'mAR_1', 'mAR_10', 'mAR_100', 'mAR_small', 'mAR_medium', 'mAR_large'
        ]}

    return avg_loss, metrics


def create_coco_ground_truth(dataset, annotation_file):
    """Create COCO ground truth object from dataset"""
    # Load the merged annotation file
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Create COCO object
    coco_gt = COCO()
    coco_gt.dataset = coco_data
    coco_gt.createIndex()

    return coco_gt


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """체크포인트 로드"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss


def visualize_sample_predictions(model, dataset, device, class_names, num_samples=5):
    """샘플 예측 결과 시각화"""
    model.eval()

    # 랜덤하게 샘플 선택
    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for idx in indices:
            image, target = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            prediction = model(image_tensor)[0]

            # CPU로 이동
            prediction = {k: v.cpu() for k, v in prediction.items()}
            target = {k: v.cpu() for k, v in target.items()}

            visualize_prediction(image, prediction, class_names, target)


def main():
    args = Args()

    # WandB 초기화
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Using device: {args.device}")

    # 어노테이션 파일 병합 (존재하지 않는 경우)
    if not os.path.exists(args.merged_annotation_path):
        print("Merging annotation files...")
        merge_annotations(args.annotation_dir, args.merged_annotation_path)

    # 데이터셋 생성
    print("Loading dataset...")
    full_dataset = CustomCocoDataset(
        image_dir=args.image_dir,
        annotation_file=args.merged_annotation_path,
        transforms=get_transforms(train=True)
    )

    # 클래스 수 설정
    args.num_classes = full_dataset.num_total_classes
    print(f"Number of classes (including background): {args.num_classes}")
    print(f"Class mapping: {full_dataset.sequential_label_to_original_name}")

    # 데이터셋 분할
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 검증 데이터셋에는 다른 transform 적용
    val_dataset.dataset.transforms = get_transforms(train=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    with open(args.label2id, 'r') as f:
        label2catid = json.load(f)

    # COCO ground truth 객체 생성 (평가용)
    print("Creating COCO ground truth object...")
    coco_gt = create_coco_ground_truth(full_dataset, args.merged_annotation_path)

    # 모델 생성
    print("Creating model...")
    model = models.get_model(args.model_name, args.num_classes)
    model.to(args.device)

    # 옵티마이저 및 스케줄러 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )

    # 체크포인트 로드 (재개 학습)
    start_epoch = 0
    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path)
        start_epoch += 1

    # WandB에 모델 watch (옵션)
    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    # 학습 루프
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        # 학습
        train_loss, train_loss_components = train_one_epoch(
            model, optimizer, train_loader, args.device, epoch,
            args.print_freq, args.use_wandb
        )
        train_losses.append(train_loss)

        # 평가 (매 eval_freq 에폭마다)
        if (epoch + 1) % args.eval_freq == 0:
            val_loss, coco_metrics = evaluate_with_coco_metrics(
                model, val_loader, args.device, coco_gt, label2catid, args.confidence_threshold
            )
            val_losses.append(val_loss)

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"mAP@0.5:0.95: {coco_metrics['mAP_0.5:0.95']:.4f}")
            print(f"mAP@0.5: {coco_metrics['mAP_0.5']:.4f}")

            # WandB 로깅
            if args.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                }

                # 훈련 loss components 추가
                for key, value in train_loss_components.items():
                    log_dict[f'train_{key}'] = value

                # COCO metrics 추가
                for key, value in coco_metrics.items():
                    log_dict[f'val_{key}'] = value

                wandb.log(log_dict)

        # 스케줄러 업데이트
        scheduler.step()

        # 체크포인트 저장
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, val_losses[-1] if val_losses else train_loss,
                            checkpoint_path)

        print(f"Epoch {epoch + 1}/{args.num_epochs} completed\n")

    # 최종 모델 저장
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    # WandB에 모델 아티팩트 저장
    if args.use_wandb:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)

    # 학습 곡선 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        # val_losses는 eval_freq마다 기록되므로 x축 조정
        val_epochs = [i * args.eval_freq for i in range(len(val_losses))]
        plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curve_path = os.path.join(args.checkpoint_dir, 'training_curves.png')
    plt.savefig(curve_path)
    plt.show()

    # WandB에 학습 곡선 업로드
    if args.use_wandb:
        wandb.log({"training_curves": wandb.Image(curve_path)})

    # 샘플 예측 시각화
    if args.visualize_predictions:
        class_names = ['background'] + [name for name in full_dataset.sequential_label_to_original_name.values()]
        visualize_sample_predictions(
            model, val_dataset, args.device, class_names, args.vis_num_samples
        )

    # WandB 종료
    if args.use_wandb:
        wandb.finish()

    print("Training completed!")


if __name__ == "__main__":
    main()