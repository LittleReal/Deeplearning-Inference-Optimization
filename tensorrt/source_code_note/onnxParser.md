# OnnxParser

Tensorrt 已经将对应的 OnnxParser 的类都封装好了，如果所有的 op 都是 TensorRT 支持的，直接调用就可以。C++ 具体的调用方法可以参考 [onnx2tensorrt main.cpp](https://github.com/onnx/onnx-tensorrt/blob/84b5be1d6fc03564f2c0dba85a2ee75bad242c2e/main.cpp)