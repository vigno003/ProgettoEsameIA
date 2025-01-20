import torch
import joblib
import yaml  # Per leggere il file di configurazione
from src.model import MeteoModel
from src.data_processing import load_and_process_data, create_dataloader
from src.train_evaluate import train_model, evaluate_model, test_model
from src.predict import prevedi_fenomeno
from src.create_graphs import create_graphs
import os
import signal
import sys
import shutil


# Funzione per caricare la configurazione
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def ask_to_skip(step_name):
    while True:
        response = input(f"Vuoi eseguire la parte '{step_name}'? (si/no): ").lower()
        if response in ['si', 'no']:
            return response == 'si'
        print("Risposta non valida. Scrivi 'si' o 'no'.")


def save_progress(model, scaler, optimizer, epoch, batch_idx, config):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'epoch': epoch,
        'batch_idx': batch_idx
    }
    torch.save(checkpoint, './models/checkpoint.pth')
    shutil.copyfile(config['scaler_path'], './models/copy_scaler.pkl')
    print("Progressi salvati.")


def signal_handler(sig, frame):
    save_progress(model, scaler, optimizer, epoch, batch_idx, config)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    global model, scaler, optimizer, epoch, batch_idx, config

    # Carica la configurazione
    config = load_config()

    # Chiedi se ripristinare i progressi
    if os.path.exists('./models/checkpoint.pth'):
        if ask_to_skip("ripristinare i progressi salvati"):
            checkpoint = torch.load('./models/checkpoint.pth')
            model = MeteoModel(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler = checkpoint['scaler']
            epoch = checkpoint['epoch']
            batch_idx = checkpoint['batch_idx']
            shutil.copyfile('./models/copy_scaler.pkl', config['scaler_path'])
        else:
            os.remove('./models/checkpoint.pth')
            os.remove('./models/copy_scaler.pkl')
            epoch = 0
            batch_idx = 0
    else:
        epoch = 0
        batch_idx = 0

    # Carica e processa i dati
    train_data, eval_data, test_data, scaler = load_and_process_data(config['csv_path'], config)
    train_dataloader = create_dataloader(train_data, config['batch_size'])
    eval_dataloader = create_dataloader(eval_data, config['batch_size'])
    test_dataloader = create_dataloader(test_data, config['batch_size'])

    if os.path.exists(config['model_path']):
        model = torch.load(config['model_path'], weights_only=False)
    else:
        model = MeteoModel(config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    if os.path.exists(config['state_dict_path']):
        model.load_state_dict(torch.load(config['state_dict_path'], weights_only=True))

    if os.path.exists(config['scaler_path']):
        scaler = joblib.load(config['scaler_path'])

    if ask_to_skip('addestramento'):
        print(f"Inizio dell'addestramento da epoch {epoch}, batch {batch_idx}...")
        train_model(model, train_dataloader, optimizer, criterion, config['num_epochs'], config, start_epoch=epoch, start_batch_idx=batch_idx)

    if ask_to_skip('valutazione'):
        print("Valutazione del modello...")
        eval_loss, eval_accuracy = evaluate_model(model, eval_dataloader, criterion)
        print(f'Evaluation Loss: {eval_loss}, Accuracy: {eval_accuracy}%')

    if ask_to_skip('test'):
        print("Test del modello...")
        test_loss, test_accuracy = test_model(model, test_dataloader, criterion)
        print(f'Test Loss: {test_loss}, Accuracy: {test_accuracy}%')

    if ask_to_skip('salvataggio'):
        torch.save(model, config['model_path'])
        torch.save(model.state_dict(), config['state_dict_path'])
        joblib.dump(scaler, config['scaler_path'])

    if ask_to_skip('creazione grafici'):
        create_graphs(config['log_dir'])

    if ask_to_skip('predizione meteo'):
        fenomeno_predetto = prevedi_fenomeno(model, scaler)
        if fenomeno_predetto:
            print(f'Il fenomeno meteo previsto è: {fenomeno_predetto}')


if __name__ == "__main__":
    main()
