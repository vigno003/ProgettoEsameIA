import torch
from torch.utils.tensorboard import SummaryWriter

def train_model(model, dataloader, optimizer, criterion, num_epochs, config):
    writer = SummaryWriter(config['log_dir'])
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct_predictions += (predicted == labels_max).sum().item()
            total_predictions += labels.size(0)

            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions * 100

            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
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
