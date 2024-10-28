import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from dataset import EiderDuckDataset
from tqdm import tqdm
import torch.optim as optim

# Hyperparameters
learning_rate = 0.001
batch_size = 4
num_epochs = 25

# Load dataset
train_dataset = EiderDuckDataset('processed-data')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = UNet(in_channels=3, num_classes=2, base_channels=128)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
    for images, density_maps, _ in progress_bar:
        images, density_maps = images.to(device), density_maps.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, density_maps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Save the model checkpoint
final_loss = running_loss/len(train_loader)
torch.save(model.state_dict(), f'unet_epoch_{epoch+1}_loss_{final_loss:6f}.pth')
