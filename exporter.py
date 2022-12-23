import os
import torch
from model.STNLPRNet import build_stnlprnet
from model.LPRNet import build_lprnet
from data.load_data import CHARS
import argparse

def Parser():
    parser = argparse.ArgumentParser(description='export lprnet model')
    parser.add_argument('--model_path', type=str, required=True, help='path to the model to be exported')
    return parser.parse_args()


def torch2onnx(modelPathNoExt, model, device):
    input_size = (1, 3, 24, 94)
    inputs = torch.randn(input_size).to(device)
    onnx_path = modelPathNoExt + ".onnx"
    torch.onnx.export(model, inputs, onnx_path,
                      verbose=True, dynamic_axes=None, opset_version=16, input_names=['input_0'], output_names=['output_0'])
    return onnx_path


if __name__ == "__main__":

    args = Parser()
    modelPath = args.model_path

    modelPathNoExt, _ = os.path.splitext(modelPath)

    # model = build_stnlprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, batch_size=1)
    model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    model.load_state_dict(torch.load(modelPath))

    print(model)

    onnx_path = torch2onnx(modelPathNoExt, model, device)
