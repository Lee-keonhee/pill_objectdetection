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
from dataset import CustomCocoDataset, custom_collate_fn, merge_annotations
from models import get_model, CustomFasterRCNN
from utils import visualize_prediction, get_id2name_dict

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우
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
        self.num_epochs = 11
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
        # 중간에 학습 멈추고 다시 시작할때, checkpoint 불러오는 설정

        self.resume = False
        self.checkpoint_path = None
        # self.resume = True
        # self.checkpoint_path = "./checkpoints/checkpoint_epoch_10.pth"
        
        # Validation
        self.val_split = 0.2  # 20% for validation
        
        # Visualization
        self.visualize_predictions = True
        self.vis_num_samples = 5


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


def main():
    args = Args()
    
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
    
    # 모델 생성
    # print("Creating model...")
    # model = CustomFasterRCNN(num_classes=args.num_classes)
    # model.to(args.device)
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
                args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} completed\n")
    
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
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curves.png'))
    plt.show()
    
    # 샘플 예측 시각화
    if args.visualize_predictions:
        class_names = ['background'] + [name for name in full_dataset.sequential_label_to_original_name.values()]
        visualize_sample_predictions(
            model, val_dataset, args.device, class_names, args.vis_num_samples
        )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
