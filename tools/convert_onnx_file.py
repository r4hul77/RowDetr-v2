import onnx
import numpy as np
from onnx import TensorProto
import copy

def fp32_to_fp16_array(fp32_array):
    """
    Converts a numpy float32 array to float16.
    """
    return fp32_array.astype(np.float16)

def convert_fp32_to_fp16(onnx_model_path, output_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    print(f"Converting model '{onnx_model_path}' from FP32 to FP16...")

    # Deep copy the model to preserve the original
    model_fp16 = copy.deepcopy(model)

    # Convert all initializers (weights/constants)
    for initializer in model_fp16.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT:  # Check for FP32
            print(f"Converting initializer '{initializer.name}' to FP16")
            # Convert raw data to FP16
            raw_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
            fp16_data = fp32_to_fp16_array(raw_data)
            initializer.raw_data = fp16_data.tobytes()
            initializer.data_type = TensorProto.FLOAT16

    # Convert input tensor types to FP16
    for input_tensor in model_fp16.graph.input:
        tensor_type = input_tensor.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT:
            print(f"Converting input '{input_tensor.name}' to FP16")
            tensor_type.elem_type = TensorProto.FLOAT16

    # Convert output tensor types to FP16
    for output_tensor in model_fp16.graph.output:
        tensor_type = output_tensor.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT:
            print(f"Converting output '{output_tensor.name}' to FP16")
            tensor_type.elem_type = TensorProto.FLOAT16

    # Save the FP16 model
    onnx.save(model_fp16, output_model_path)
    print(f"FP16 model saved to '{output_model_path}'")

if __name__ == "__main__":
    # Input and output paths for the ONNX model
    input_model_path = "modified_efficient_f15_new.onnx"  # Replace with your input model path
    output_model_path = "modified_efficient_f15_new_efficient_F1_fp16.onnx"  # Replace with desired output path

    convert_fp32_to_fp16(input_model_path, output_model_path)
