# Plugin
在 6.0.1 之前的TensorRT版本中，从 IPluginV2 或 IPluginV2Ext 派生自定义层的. 6.0.1版本及之后的版本仍支持这些API，但强烈建议转到 IPluginV2IOExt 或 IPluginV2DynamicExt，以便能够使用新的 plugin 功能.

## 定义 Plugin Layer
**继承 IPluginV2 或 IPluginV2Ext、IPluginV2IOExt、IPluginV2DynamicExt**，实现自定义插件的基类，包含版本化和对其它格式和单精度的处理。  
**继承 IPluginCreator**，自定义层的创建类，可以通过它获取插件的名称、版本信息、参数等，也提供创建插件的方法，并在推理阶段反序列化它。  
**REGISTER_TENSORRT_PLUGIN(pluginCreator)** 进行静态注册定义好的 plugin，并在使用时通过 getPluginRegistry() 查询并使用。官方已经实现的插件有：  
RPROI_TRT  
Normalize_TRT  
PriorBox_TRT  
GridAnchor_TRT  
NMS_TRT  
LReLU_TRT  
Reorg_TRT  
Region_TRT  
Clip_TRT  

## Network Add Plugin
使用extern函数getPluginRegistry访问全局TensorRT插件注册表，然后访问当前需要的 plugin 的 creator  
`auto creator = getPluginRegistry()->getPluginCreator(pluginName, pluginVersion);`  
`const PluginFieldCollection* pluginFC = creator->getFieldNames();`  
填充该层参数信息，pluginData 需要先通过 PluginField 分配堆上空间 
`PluginFieldCollection* pluginData = parseAndFillFields(pluginFC, layerFields);`  
创建 plugin  
`IPluginV2 *pluginObj = creator->createPlugin(layerName, pluginData);`  
将 plugin 添加到 network 中  
`auto layer = network.addPluginV2(&inputs[0], int(inputs.size()), pluginObj);`  
添加其他层，序列化 engine 等  
释放对象或数据资源  
`pluginObj->destroy()`  
… (destroy network, engine, builder)  
… (free allocated pluginData)  
