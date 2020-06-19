# Keras Convert to Tensorrt By Onnx

## Keras Convert to Onnx
`python keras_to_onnx.py --keras_path keras_model_path --onnx_path onnx_model_save_path`  

注意 `keras2onnx` 的要从源码安装：  
pip install -U git+https://github.com/microsoft/onnxconverter-common
pip install -U git+https://github.com/onnx/keras-onnx  
refer to [keras2onnx](https://github.com/onnx/keras-onnx)

## Onnx Convert to Tensorrt
参考 [onnx2trt](../../onnx_to_tensorrt/README.md)

## Bug Needed to be Fixed
转化到 TensorRT 后，速度有近 5 倍的提升，但是显存也飙升，相同的模型如果直接从 Tensorrt 中手动创建 Network 的方式来构建，显存和通过 onnx-tensorrt 的方式相比相差不大，具体原因还没有找到。