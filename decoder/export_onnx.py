"""Export the trained RPC-Net to ONNX (e.g. to inspect it with Netron).

Usage:
    python -m decoder.export_onnx
"""
import torch

import config
from decoder.rpc_net import ANGLE_INPUT_SIZE, EMG_INPUT_SIZE, RPCNet


def export_to_onnx():
    model = RPCNet()
    if config.RPC_NET_WEIGHTS.exists():
        print(f"Loading trained weights from {config.RPC_NET_WEIGHTS.name}")
        model.load_state_dict(torch.load(config.RPC_NET_WEIGHTS, map_location="cpu", weights_only=True))
    else:
        print("Weights not found: exporting the architecture with random weights.")
    model.eval()

    dummy_emg = torch.randn(1, EMG_INPUT_SIZE)
    dummy_ang = torch.randn(1, ANGLE_INPUT_SIZE)
    torch.onnx.export(
        model,
        (dummy_emg, dummy_ang),
        str(config.RPC_NET_ONNX),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_emg", "input_ang"],
        output_names=["output_kinematics"],
        dynamic_axes={"input_emg": {0: "batch_size"},
                      "input_ang": {0: "batch_size"},
                      "output_kinematics": {0: "batch_size"}},
    )
    print(f"ONNX model saved to {config.rel(config.RPC_NET_ONNX)}")


if __name__ == "__main__":
    export_to_onnx()
