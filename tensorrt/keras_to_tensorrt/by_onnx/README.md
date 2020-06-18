# Keras Convert to Tensorrt By Onnx

## Keras Convert to Onnx
`python keras2onnx.py --keras_path keras_model_path --input_name model_input_name --output_name model_output_name --onnx_path onnx_model_save_path`  
`input_name`、`output_name` 用于将 onnx_model 中的 batch 维改为 1，因为在接下来的 onnx-tensorrt 中不支持 batch 维为不确定的 `N`。

## Onnx Convert to Tensorrt
参考 [onnx2trt](../../onnx_to_tensorrt/README.md)

## Bug Needed to be Fixed
转化到 TensorRT 后，速度有近 5 倍的提升，但是显存也飙升，相同的模型如果直接从 Tensorrt 中手动创建 Network 的方式来构建，显存和通过 onnx-tensorrt 的方式相比相差不大，具体原因还没有找到。