import tensorrt as trt
import numpy as np
import common
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description = 'Parameters for keras model convert to onnx')
    parser.add_argument('--onnx_path', '-onnx', required = True, type = str, 
    help = 'onnx model file path')
    parser.add_argument('--trt_path', '-trt', required = True, type = str, 
    help = 'save path of tensorrt model')
    args = parser.parse_args()

    model_path = args.onnx_path

    # VERBOSE 可以查看转化过程中各个层的输入输出信息
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    model = open(model_path, 'rb')

    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, builder.create_builder_config())
    
    plan=engine.serialize()
    with open(args.trt_path, "wb") as f:
        f.write(plan)
    f.close()

if __name__ == "__main__":
    main()