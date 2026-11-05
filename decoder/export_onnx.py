import torch
import os
from RPC_Net import RPCNet_Exact

def export_to_onnx():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "rpc_net_weights.pth")
    onnx_path = os.path.join(BASE_DIR, "rpc_net.onnx")

    print("Inizializzazione del modello RPCNet_Exact...")
    model = RPCNet_Exact(in_emg=512, in_ang=192)

    # Carica i pesi se esistono (opzionale per Netron, ma buona pratica)
    if os.path.exists(model_path):
        print(f"Caricamento dei pesi addestrati da: {os.path.basename(model_path)}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    else:
        print("Pesi non trovati. Verrà esportata l'architettura con pesi casuali (sufficiente per Netron).")

    model.eval()

    # Tensori "dummy" per simulare un batch di inferenza (Batch Size = 1)
    dummy_emg = torch.randn(1, 512)
    dummy_ang = torch.randn(1, 192)

    print(f"Esportazione del modello ONNX in corso...")
    torch.onnx.export(
        model,                                      # Modello da esportare
        (dummy_emg, dummy_ang),                     # Input fittizi passati al forward
        onnx_path,                                  # Percorso di salvataggio
        export_params=True,                         # Salva i pesi allenati assieme alla struttura
        opset_version=11,                           # Versione ONNX stabile
        do_constant_folding=True,                   # Ottimizza i calcoli costanti
        input_names=['input_emg', 'input_ang'],     # Nomi degli input (visibili su Netron)
        output_names=['output_kinematics'],         # Nomi dell'output
        dynamic_axes={'input_emg': {0: 'batch_size'}, 'input_ang': {0: 'batch_size'}, 'output_kinematics': {0: 'batch_size'}}
    )
    print(f"Esportazione completata con successo! File salvato: {os.path.basename(onnx_path)}")

if __name__ == "__main__":
    export_to_onnx()