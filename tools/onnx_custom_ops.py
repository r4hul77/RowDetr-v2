import onnxscript
from onnxscript.onnx_opset import opset16 as op
import torch
import numpy as np
opset_version = 16
custom_opset = onnxscript.values.Opset(domain="com.test", version=1)

@onnxscript.script(custom_opset)
def SortedSearch(L, idxs):
    # Efficient implementation equivalent to the following:
    l1 = op.Unsqueeze(L,[1, 2])
    idxs = op.Unsqueeze(idxs, [0])
    vals = op.Size(L) - op.ReduceSum(op.Cast((l1>=idxs), 7), [0], False)
    # vals = op.Cast(op.Shape(l1), 7)
    return op.Cast(vals, 7)

def custom_sorted_search(g, L, idxs, out_int32=False, right=False, side=None, out=None, sorter=None):
    return g.onnxscript_op(SortedSearch, L, idxs)

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::searchsorted",
    symbolic_fn=custom_sorted_search,
    opset_version=opset_version,
)

