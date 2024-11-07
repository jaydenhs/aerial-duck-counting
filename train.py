import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from dataset import EiderDuckDataset
from tqdm import tqdm
import datetime
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Hyperparameters
learning_rate = 0.001
batch_size = 2
num_epochs = 150

# Load dataset
train_dataset = EiderDuckDataset('processed-data')
def collate_fn(batch):
    images, density_maps, *_ = zip(*batch)
    images = torch.stack(images, 0)
    density_maps = torch.stack(density_maps, 0)
    return images, density_maps

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize model, loss function, and optimizer
model = UNet(in_channels=3, num_classes=2, base_channels=128)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
    for images, density_maps, *_ in progress_bar:
        images, density_maps = images.to(device), density_maps.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, density_maps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
    
    # Step the scheduler
    scheduler.step()

# Save the model checkpoint
final_loss = running_loss/len(train_loader)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
torch.save(model.state_dict(), f'checkpoints/{timestamp}_unet_epoch_{epoch+1}_loss_{final_loss:6f}.pth')
