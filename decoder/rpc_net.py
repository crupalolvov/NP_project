"""RPC-Net: regression of continuous hand kinematics from EMG.

Re-implementation of the architecture in Fig. 3 of the RPC-Net paper: one
independent sub-network per degree of freedom (24 in total). Each sub-network
has two input branches whose embeddings are concatenated:

* EMG branch    - 512 inputs (16 time steps x 32 RMS channels) -> FC 512
* Angle branch  - 192 inputs (8 past time steps x 24 DoF)      -> 3 x FC 24
* Merged head   - 3 x FC 134 -> FC 1 -> Sigmoid (normalized angle in [0, 1])
"""
import torch
import torch.nn as nn

EMG_INPUT_SIZE = 512
ANGLE_INPUT_SIZE = 192
N_DOF = 24


class SingleJointNet(nn.Module):
    """Predicts one normalized joint angle from EMG history and past angles."""

    def __init__(self, in_emg=EMG_INPUT_SIZE, in_ang=ANGLE_INPUT_SIZE):
        super().__init__()
        self.emg_branch = nn.Sequential(
            nn.Linear(in_emg, 512), nn.ReLU(),
        )
        self.ang_branch = nn.Sequential(
            nn.Linear(in_ang, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
        )
        self.merged = nn.Sequential(
            nn.Linear(512 + 24, 134), nn.ReLU(),
            nn.Linear(134, 134), nn.ReLU(),
            nn.Linear(134, 134), nn.ReLU(),
            nn.Linear(134, 1),     # final FC without ReLU
            nn.Sigmoid(),
        )

    def forward(self, emg, ang):
        combined = torch.cat((self.emg_branch(emg), self.ang_branch(ang)), dim=1)
        return self.merged(combined)


class RPCNet(nn.Module):
    """24 parallel ``SingleJointNet`` - one per degree of freedom."""

    def __init__(self, in_emg=EMG_INPUT_SIZE, in_ang=ANGLE_INPUT_SIZE, n_dof=N_DOF):
        super().__init__()
        self.sub_nets = nn.ModuleList([SingleJointNet(in_emg, in_ang) for _ in range(n_dof)])

    def forward(self, emg, ang):
        return torch.cat([net(emg, ang) for net in self.sub_nets], dim=1)


# Optimizer settings reported in the paper (used when no tuned values exist)
PAPER_HYPERPARAMETERS = {"lr": 1e-4, "eps": 1e-3, "weight_decay": 0.0, "batch_size": 10}
ADAM_BETAS = (0.9, 0.99)
