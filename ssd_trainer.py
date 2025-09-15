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
from dataset import CustomCocoDataset, custom_collate_fn
from models import get_model, CustomFasterRCNN
from utils import visualize_prediction, get_id2name_dict

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우
plt.rcParams['axes.unicode_minus'] = False


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


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """한 에폭 학습 함수"""
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if batch_idx % print_freq == 0:
            print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, data_loader, device):
    """검증 함수"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 평가 모드에서는 targets를 전달하지 않음
            model.train()  # loss 계산을 위해 일시적으로 train 모드
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            model.eval()  # 다시 eval 모드로

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


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


def main(args):
    print(f'사용중인 모델 명 : {args.model_name}')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)

    # 체크포인트 디렉토리 생성(존재하지 않을 경우)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Using device: {args.device}")

    # 어노테이션 파일 존재 확인
    if not os.path.exists(args.train_annotation_path):
        raise FileNotFoundError(f"Training annotation file not found: {args.train_annotation_path}")
    if not os.path.exists(args.valid_annotation_path):
        raise FileNotFoundError(f"Validation annotation file not found: {args.valid_annotation_path}")

    # 데이터셋 생성 - 훈련용과 검증용을 별도로 생성
    print("Loading datasets...")
    train_dataset = CustomCocoDataset(
        image_dir=args.train_image_dir,
        annotation_file=args.train_annotation_path,
        transforms=get_transforms(train=True)
    )

    val_dataset = CustomCocoDataset(
        image_dir=args.valid_image_dir,
        annotation_file=args.valid_annotation_path,
        transforms=get_transforms(train=False)
    )

    # 클래스 수 설정 (훈련 데이터셋 기준)
    args.num_classes = train_dataset.num_total_classes
    print(f"Number of classes (including background): {args.num_classes}")
    print(f"Class mapping: {train_dataset.sequential_label_to_original_name}")

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

    # 학습 루프
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        # 학습
        train_loss = train_one_epoch(
            model, optimizer, train_loader, args.device, epoch, args.print_freq
        )
        train_losses.append(train_loss)

        # 검증
        val_loss = evaluate(model, val_loader, args.device)
        val_losses.append(val_loss)

        # 스케줄러 업데이트
        scheduler.step()

        # 체크포인트 저장
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)

        print(f"Epoch {epoch + 1}/{args.num_epochs} completed\n")

    # 최종 모델 저장
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    # 학습 곡선 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, args.model_name, 'training_curves.png'))
    plt.show()

    # 샘플 예측 시각화
    if args.visualize_predictions:
        class_names = ['background'] + [name for name in train_dataset.sequential_label_to_original_name.values()]
        visualize_sample_predictions(
            model, val_dataset, args.device, class_names, args.vis_num_samples
        )

    print("Training completed!")


if __name__ == "__main__":
    args = Args()
    main(args)