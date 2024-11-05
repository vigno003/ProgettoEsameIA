import torch
from torch.utils.tensorboard import SummaryWriter

def train_model(model, dataloader, optimizer, criterion, num_epochs, log_dir='runs/meteo_model'):
    writer = SummaryWriter(log_dir)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

    writer.close()

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)
