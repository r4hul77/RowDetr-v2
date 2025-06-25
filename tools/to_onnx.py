import mmcv
import torch
import os
import argparse
from models.poly_detection import PolyNet
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

def convert2Onnx(model, input_size, output_path):
    input_shape = (1, 3, input_size[0], input_size[1])
    model.eval()
    model.cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        dummy_input = torch.randn(input_shape).cuda()
        # # Export the model
        m = copy.deepcopy(model)
        #m = quantize_fx.fuse_fx(m)
        # qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
        # m = quantize_fx.prepare_fx(m, qconfig_mapping, (dummy_input.detach()))
        # m = quantize_fx.convert_fx(m)
        torch.onnx.export(
            m,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            output_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            do_constant_folding=True,  # whether to execute constant folding for optimization
            opset_version=17,
            verbose=True
        )
    # torch.onnx.export(model, dummy_input, output_path, verbose=False)


from mmengine import load
def checkpoint2Torch(path, mm):
    checkpoint = torch.load(path)
    mm.load_state_dict(checkpoint['state_dict'])
    return mm

