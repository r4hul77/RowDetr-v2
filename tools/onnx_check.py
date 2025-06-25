
import onnx
from onnx import TensorProto

# List of TensorRT-supported data types
TENSORRT_SUPPORTED_DTYPES = {
    # TensorProto.FLOAT,   # FP32
    TensorProto.FLOAT16, # FP16
    # TensorProto.UINT8,   # INT8 (Quantized)
}

# Map ONNX data type enums to readable strings
ONNX_DTYPE_MAP = {
    # TensorProto.FLOAT: "FLOAT (FP32)",
    TensorProto.FLOAT16: "FLOAT16 (FP16)",
    TensorProto.INT32: "INT32",
    TensorProto.INT64: "INT64",
    TensorProto.BOOL: "BOOL",
    TensorProto.UINT8: "UINT8 (INT8)",
    TensorProto.DOUBLE: "DOUBLE",
}

def check_unsupported_datatypes(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    print(f"Checking unsupported datatypes in model: {onnx_model_path}\n")
    unsupported_tensors = []

    # Check all inputs
    for inp in model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        if elem_type not in TENSORRT_SUPPORTED_DTYPES:
            unsupported_tensors.append((inp.name, ONNX_DTYPE_MAP.get(elem_type, f"UNKNOWN ({elem_type})")))

    # Check all outputs
    for out in model.graph.output:
        elem_type = out.type.tensor_type.elem_type
        if elem_type not in TENSORRT_SUPPORTED_DTYPES:
            unsupported_tensors.append((out.name, ONNX_DTYPE_MAP.get(elem_type, f"UNKNOWN ({elem_type})")))

    # Check all initializers (intermediate tensors/constants)
    for init in model.graph.initializer:
        elem_type = init.data_type
        print(elem_type)
        if elem_type not in TENSORRT_SUPPORTED_DTYPES:
            unsupported_tensors.append((init.name, ONNX_DTYPE_MAP.get(elem_type, f"UNKNOWN ({elem_type})")))

    # Report results
    if unsupported_tensors:
        print("Found unsupported datatypes:")
        for name, dtype in unsupported_tensors:
            print(f" - Tensor Name: {name}, DataType: {dtype}")
    else:
        print("All tensors use supported datatypes for TensorRT!")

    print("\nDatatype check complete.")

if __name__ == "__main__":
    # Path to your ONNX model
    
    model_path = "modified_efficient_f15_new.onnx"
    check_unsupported_datatypes(model_path)

