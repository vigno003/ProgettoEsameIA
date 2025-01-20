import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_model(model, train_dataloader, optimizer, criterion, num_epochs, config, start_epoch=0, start_batch_idx=0):
    writer = SummaryWriter(config['log_dir'])
    model.train()
    all_labels = []
    all_predictions = []
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        print(f"Epoch {epoch+1}/{num_epochs} (ripartendo da batch {start_batch_idx})")
        for batch_idx, (inputs, labels) in enumerate(train_dataloader, start=start_batch_idx):
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

            all_labels.extend(labels_max.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions * 100

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {accuracy}%')

        start_batch_idx = 0  # Reset batch_idx at the end of each epoch

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    writer.close()

def evaluate_model(model, eval_dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in eval_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct_predictions += (predicted == labels_max).sum().item()
            total_predictions += labels.size(0)

            all_labels.extend(labels_max.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(eval_dataloader)
    accuracy = correct_predictions / total_predictions * 100
    print(f'Evaluation Loss: {avg_loss}, Accuracy: {accuracy}%')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    return avg_loss, accuracy

def test_model(model, test_dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct_predictions += (predicted == labels_max).sum().item()
            total_predictions += labels.size(0)

            all_labels.extend(labels_max.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions * 100
    print(f'Test Loss: {avg_loss}, Accuracy: {accuracy}%')

    return avg_loss, accuracy
