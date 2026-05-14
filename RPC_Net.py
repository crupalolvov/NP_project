import torch
import torch.nn as nn
import torch.optim as optim

class SingleJointNet_Exact(nn.Module):
    """Architettura esatta estratta dalla Fig. 3 del paper RPC-Net"""
    def __init__(self, in_emg=512, in_ang=192):
        super().__init__()
        
        # Ramo EMG: 1 strato (ReLU and FC)
        self.emg_branch = nn.Sequential(
            nn.Linear(in_emg, 512),
            nn.ReLU()
        )
        
        # Ramo Angoli: 3 strati (ReLU and FC)
        self.ang_branch = nn.Sequential(
            nn.Linear(in_ang, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU()
        )
        
        # Ramo Convergente: Concatenazione + 3 strati + Output
        self.merged = nn.Sequential(
            nn.Linear(512 + 24, 134),
            nn.ReLU(),
            nn.Linear(134, 134),
            nn.ReLU(),
            nn.Linear(134, 134),
            nn.ReLU(),
            nn.Linear(134, 1)  # FC finale senza ReLU
        )

    def forward(self, emg, ang):
        e_out = self.emg_branch(emg)
        a_out = self.ang_branch(ang)
        combined = torch.cat((e_out, a_out), dim=1)
        return self.merged(combined)

class RPCNet_Exact(nn.Module):
    def __init__(self, in_emg=512, in_ang=192):
        super().__init__()
        self.sub_nets = nn.ModuleList([SingleJointNet_Exact(in_emg, in_ang) for _ in range(24)])

    def forward(self, emg, ang):
        outputs = [net(emg, ang) for net in self.sub_nets]
        return torch.cat(outputs, dim=1)

# --- CONFIGURAZIONE ESATTA DELL'OTTIMIZZATORE ---
# Da usare nel tuo ciclo di training al posto della Grid Search
model = RPCNet_Exact()
criterion = nn.MSELoss() #

# Iperparametri hard-coded presi dal paper
optimizer = optim.Adam(
    model.parameters(), 
    lr=1e-5, 
    eps=1e-3, 
    betas=(0.9, 0.99)
)

batch_size = 10  #
epochs = 3       #