import json
import os
import torch
import sys
from model.STNLPRNet import build_stnlprnet
from data.load_data import CHARS

def help():
    print("exporter.py <model_path> <exported_model_type>")


def torch2onnx(modelPathNoExt, model, device):
    input_size = (1, 3, 24, 94)
    inputs = torch.randn(input_size).to(device)
    onnx_path = modelPathNoExt + ".onnx"
    torch.onnx.export(model, inputs, onnx_path,
                      verbose=True, dynamic_axes=None, opset_version=16, input_names=['input_0'], output_names=['output_0'])
    return onnx_path


if __name__ == "__main__":
    print("exporter")
    if (len(sys.argv) < 3):
        help()
        sys.exit()
    modelPath = sys.argv[1]
    modelDst = sys.argv[2]

    modelPathNoExt, _ = os.path.splitext(modelPath)

    model = build_stnlprnet(lpr_max_len=8, phase=False,
                          class_num=len(CHARS), dropout_rate=0, batch_size=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    model.load_state_dict(torch.load(modelPath))

    print(model)

    if modelDst == "onnx":
        onnx_path = torch2onnx(modelPathNoExt, model, device)
