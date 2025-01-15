import torch
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_dataloader, optimizer, criterion, num_epochs, config):
    writer = SummaryWriter(config['log_dir'])
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_dataloader:
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

        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions * 100

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {accuracy}%')

    writer.close()

def evaluate_model(model, eval_dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in eval_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct_predictions += (predicted == labels_max).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(eval_dataloader)
    accuracy = correct_predictions / total_predictions * 100
    print(f'Evaluation Loss: {avg_loss}, Accuracy: {accuracy}%')
    return avg_loss, accuracy

def test_model(model, test_dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct_predictions += (predicted == labels_max).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions * 100
    print(f'Test Loss: {avg_loss}, Accuracy: {accuracy}%')
    return avg_loss, accuracy
