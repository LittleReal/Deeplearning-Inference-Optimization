import keras
import keras2onnx
import onnx
from keras.models import load_model
# import onnxruntime
import numpy as np
import argparse

def change_batch_dim(model, actual_batch=1):
    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        if dim1.dim_param=="N":
            dim1.dim_value = actual_batch

    outputs = model.graph.output
    for output in outputs:
        dim1 = output.type.tensor_type.shape.dim[0]
        if dim1.dim_param=="N":
            dim1.dim_value = actual_batch

def main():
    parser = argparse.ArgumentParser(description = 'Parameters for keras model convert to onnx')
    parser.add_argument('--keras_path', '-keras', required = True, type = str, 
    help = 'keras model file path')
    parser.add_argument('--onnx_path', '-onnx', required = True, type = str, 
    help = 'save path of onnx model')
    parser.add_argument("--batch_size", "-batch", type = int, default = 1,
    help = "actual batch size")
    args = parser.parse_args()

    keras_model = load_model(args.keras_path)
    # target_opset sometimes needs to be given
    onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
    change_batch_dim(onnx_model, args.batch_size)
    onnx.save_model(onnx_model, args.onnx_path)

    # onnx_model inference.
    # x = np.ones((1, 94, 39, 1), dtype=np.float32)
    # content = onnx_model.SerializeToString()
    # sess = onnxruntime.InferenceSession(content)
    # input_name = sess.get_inputs()[0].name
    # pred_onnx = sess.run(None, {input_name: x})

if __name__ == "__main__":
    main()
