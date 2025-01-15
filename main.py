import torch
import joblib
import yaml  # Per leggere il file di configurazione
from src.model import MeteoModel
from src.data_processing import load_and_process_data, create_dataloader
from src.train_evaluate import train_model, evaluate_model
from src.predict import prevedi_fenomeno
from src.create_graphs import create_graphs
import os


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


def main():
    # Carica la configurazione
    config = load_config()

    # Carica e processa i dati
    train_data, eval_data, scaler = load_and_process_data(config['csv_path'], config)
    train_dataloader = create_dataloader(train_data, config['batch_size'])
    eval_dataloader = create_dataloader(eval_data, config['batch_size'])

    if os.path.exists(config['model_path']):
        model = torch.load(config['model_path'], weights_only=False)
    else:
        model = MeteoModel(config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    if os.path.exists(config['state_dict_path']):
        model.load_state_dict(torch.load(config['state_dict_path']))

    if os.path.exists(config['scaler_path']):
        scaler = joblib.load(config['scaler_path'])

    if ask_to_skip('addestramento'):
        print("Inizio dell'addestramento...")
        train_model(model, train_dataloader, optimizer, criterion, config['num_epochs'], config)

    if ask_to_skip('valutazione'):
        print("Valutazione del modello...")
        test_loss = evaluate_model(model, eval_dataloader, criterion)
        print(f'Test Loss: {test_loss}')

    if ask_to_skip('salvataggio'):
        torch.save(model, config['model_path'])
        torch.save(model.state_dict(), config['state_dict_path'])
        joblib.dump(scaler, config['scaler_path'])

    if ask_to_skip('creazione grafici'):
        create_graphs(config)

    if ask_to_skip('predizione meteo'):
        fenomeno_predetto = prevedi_fenomeno(model, scaler)
        if fenomeno_predetto:
            print(f'Il fenomeno meteo previsto è: {fenomeno_predetto}')


if __name__ == "__main__":
    main()
