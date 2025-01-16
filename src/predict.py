import torch
import pandas as pd

def prevedi_fenomeno(model, scaler):
    try:
        # Input dell'utente
        temperatura = float(input("Inserisci la temperatura (°C): "))
        umidita = float(input("Inserisci l'umidità (%): "))
        vento = float(input("Inserisci la velocità del vento (km/h): "))
        pressione = float(input("Inserisci la pressione (mb): "))

        # Creazione del DataFrame con i nomi delle colonne
        dati = pd.DataFrame([[temperatura, umidita, vento, pressione]], columns=['temperatura', 'umidita', 'vento', 'pressione'])

        # Normalizzazione dei dati
        dati_normalizzati = scaler.transform(dati)
        input_tensor = torch.tensor(dati_normalizzati, dtype=torch.float32).view(1, -1)

        # Debug: stampa i dati normalizzati
        print("Dati normalizzati:", dati_normalizzati)

        # Previsione
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            predizione = torch.argmax(output, dim=1)

        # Debug: stampa l'output del modello
        print("Output del modello:", output)
        print("Predizione:", predizione)

        fenomeni = ["Sole", "Pioggia", "Nebbia"]
        return fenomeni[predizione.item()]

    except Exception as e:
        print(f"Errore durante la previsione: {e}")
        return None
