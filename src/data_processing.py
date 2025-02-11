import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch


def load_and_process_data(csv_path, config):
    # Caricamento dati
    csv = pd.read_csv(csv_path, sep=';')
    csv['FENOMENI'] = csv['FENOMENI'].fillna('sole').str.strip()

    # Seleziona le colonne necessarie
    data = csv[['TMEDIA °C', 'UMIDITA %', 'VENTOMEDIA km/h', 'PRESSIONESLM mb', 'FENOMENI']]
    data.columns = ['temperatura', 'umidita', 'vento', 'pressione', 'fenomeni']

    # Rimuove righe con valori mancanti
    data = data.dropna()

    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    data[['temperatura', 'umidita', 'vento', 'pressione']] = scaler.fit_transform(
        data[['temperatura', 'umidita', 'vento', 'pressione']])

    # Codifica delle variabili categoriali (fenomeni)
    data = pd.get_dummies(data, columns=['fenomeni'], prefix='', prefix_sep='')
    data.rename(columns={'nebbia': 'fenomeni_nebbia', 'pioggia': 'fenomeni_pioggia', 'sole': 'fenomeni_sole'},
                inplace=True)

    # Conversione colonne booleane in numeriche
    data['fenomeni_nebbia'] = data.get('fenomeni_nebbia', 0).astype(int)
    data['fenomeni_pioggia'] = data.get('fenomeni_pioggia', 0).astype(int)
    data['fenomeni_sole'] = data['fenomeni_sole'].astype(int)

    # Divisione del dataset in training (60%), testing (20%) e evaluation (20%)
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, eval_data, test_data, scaler


class MeteoDataset(Dataset):
    def __init__(self, data):
        self.features = data[['temperatura', 'umidita', 'vento', 'pressione']].values
        self.targets = data[['fenomeni_sole', 'fenomeni_pioggia', 'fenomeni_nebbia']].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def create_dataloader(data, batch_size=32):
    dataset = MeteoDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return dataloader
