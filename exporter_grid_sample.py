import os
import torch
import sys
from model.StnLprNet import build_stnlprnet
from data.load_data import CHARS
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import torch.onnx.symbolic_opset11 as sym_opset
import torch.onnx.symbolic_helper as sym_help

def help():
    print("exporter.py <model_path> <exported_model_type>")

def grid_sampler(g, input, grid, mode, padding_mode, aligncorners): #long, long, long: contants dtype
    mode_i = sym_help._maybe_get_scalar(mode)
    paddingmode_i = sym_help._maybe_get_scalar(padding_mode)
    aligncorners_i = sym_help._maybe_get_scalar(aligncorners)

    return g.op("GridSample", input, grid, mode_i=mode_i, padding_mode_i=paddingmode_i,
     align_corners_i=aligncorners_i) #just a dummy definition for onnx runtime since we don't need onnx inference

sym_opset.grid_sampler = grid_sampler

def torch2onnx(modelPathNoExt, model, device):
    N = 1
    H = 24
    W = 94
    C = 3
    input_size = (N, C, H, W)
    inputs = torch.randn(input_size).to(device)
    onnx_path = modelPathNoExt + ".onnx"
    torch.onnx.export(model, inputs, onnx_path,
                      verbose=True, dynamic_axes=None, opset_version=16, input_names=['input_0'], output_names=['output_0'])

    return onnx_path

def modify_onnx(onnx_model_file):
    graph = gs.import_onnx(onnx.load(onnx_model_file))
    assert(graph is not None)

    for node in graph.nodes:
        if node.op == 'GridSample':
            print("GridSampler found")
            _, c, h, w = node.inputs[0].shape
            h_g = h
            w_g = w
            align_corners = node.attrs['align_corners']
            mode_str = node.attrs['mode']
            mode_int = ['bilinear', 'nearest', 'bicubic'].index(mode_str)
            pad_mode_str = node.attrs['padding_mode']
            pad_mode_int = ['zeros','border','reflection'].index(pad_mode_str)
            m_type = 0 if node.inputs[0].dtype == np.float32 else 1
            buffer = np.array([c, h, w, h_g, w_g], dtype=np.int64).tobytes('C') \
              + np.array([mode_int, pad_mode_int], dtype=np.int32).tobytes('C') \
              + np.array([align_corners], dtype=np.bool).tobytes('C') \
              + np.array([m_type], dtype=np.int32).tobytes('C')
            node.attrs = {'name':'GridSampler', 'version':'1', 'namespace':"", 'data':buffer}
            node.op = 'TRT_PluginV2'
    
    onnx.save(gs.export_onnx(graph), onnx_model_file)

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
    # model.float()  # load to FP32
    model.to(device).eval()
    model.load_state_dict(torch.load(modelPath))

    print(model)

    if modelDst == "onnx":
        onnx_path = torch2onnx(modelPathNoExt, model, device)
        print("exported")
        modify_onnx(onnx_path)
        print("modified")