import torch


def prevedi_fenomeno(model, scaler):
    try:
        # Input dell'utente
        temperatura = float(input("Inserisci la temperatura (°C): "))
        umidita = float(input("Inserisci l'umidità (%): "))
        vento = float(input("Inserisci la velocità del vento (km/h): "))
        pressione = float(input("Inserisci la pressione (mb): "))

        # Normalizzazione dei dati
        dati_normalizzati = scaler.transform([[temperatura, umidita, vento, pressione]])
        input_tensor = torch.tensor(dati_normalizzati, dtype=torch.float32).view(1, -1)

        # Previsione
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            predizione = torch.argmax(output, dim=1)

        fenomeni = ["Sole", "Pioggia", "Nebbia"]
        return fenomeni[predizione.item()]

    except Exception as e:
        print(f"Errore durante la previsione: {e}")
        return None