import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from syn_dataset import DuckDataset, collate_fn
from torch.utils.data import DataLoader
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Dataset and DataLoader
    dataset = DuckDataset('synthesized/', 'synthesized/annotations.json')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 3  # 2 classes (male, female) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=0.005)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-6)

    num_epochs = 500
    epoch_losses = []
    pbar = tqdm(range(num_epochs), desc=f"Training for {num_epochs} epochs")
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        epoch_losses.append(epoch_loss/len(data_loader))
        pbar.set_postfix(loss=epoch_loss/len(data_loader))
        scheduler.step(epoch_loss/len(data_loader))

    # Save the model with timestamp, epoch number, and loss
    output_dir = 'syn_checkpoints'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{output_dir}/{timestamp}_epoch{epoch+1}_loss{losses.item():.4f}.pth"
    torch.save(model.state_dict(), model_filename)

    # Plot training loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)

    # Save the plot
    plot_filename = f"{output_dir}/{timestamp}_epoch{epoch+1}_loss{losses.item():.4f}.png"
    plt.savefig(plot_filename)
    plt.close()