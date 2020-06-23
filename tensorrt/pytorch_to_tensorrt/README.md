# Pytorch Model to Tensorrt

## Pytorch 存储和加载模型方式  

### 存储和加载整个模型  
`torch.save(model_object, "model.pt")`  
`model_object = torch.load("model.pt")`

`torch.save(model_object.state_dict(), "params.pt")`  
`model_object.load_state_dict("params.pt")`

如果只是进行推理，并且模型中包含 BatchNorm 或者 Dropout，需要 `model.eval()`。因为在训练和预测时，这两个 op forward 时的参数不同。

## [By Onnx](by_onnx/README.md)
