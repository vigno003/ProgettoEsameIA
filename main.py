import torch
import joblib
import yaml  # Per leggere il file di configurazione
from src.model import MeteoModel
from src.data_processing import load_and_process_data, create_dataloader
from src.train_evaluate import train_model, evaluate_model
from src.predict import prevedi_fenomeno
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

    # Percorsi ai file
    csv_path = config['csv_path']
    model_path = config['model_path']
    state_dict_path = config['state_dict_path']
    scaler_path = config['scaler_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    train_data, eval_data, scaler = load_and_process_data(csv_path)
    train_dataloader = create_dataloader(train_data, batch_size)
    eval_dataloader = create_dataloader(eval_data, batch_size)

    if os.path.exists(model_path):
        model = torch.load(model_path, weights_only=False)
    else:
        model = MeteoModel()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path))

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    if ask_to_skip('addestramento'):
        print("Inizio dell'addestramento...")
        train_model(model, train_dataloader, optimizer, criterion, num_epochs)

    if ask_to_skip('valutazione'):
        print("Valutazione del modello...")
        test_loss = evaluate_model(model, eval_dataloader, criterion)
        print(f'Test Loss: {test_loss}')

    if ask_to_skip('salvataggio'):
        torch.save(model, model_path)
        torch.save(model.state_dict(), state_dict_path)
        joblib.dump(scaler, scaler_path)

    if ask_to_skip('predizione meteo'):
        fenomeno_predetto = prevedi_fenomeno(model, scaler)
        if fenomeno_predetto:
            print(f'Il fenomeno meteo previsto è: {fenomeno_predetto}')


if __name__ == "__main__":
    main()
