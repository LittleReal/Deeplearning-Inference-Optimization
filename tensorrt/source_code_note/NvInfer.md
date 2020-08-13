# NnInfer.h
NvInfer.h 中主要定义了一些基础类和 ILayer、INetWorkDefinition、IBuilder，以及继承自 ILayer 的 IConvolutionLayer 等，也是在 Model 转换或创建过程中参考最多的 include 文件。

## Dims
关于 Dims 相关 class 的定义，Dims2, DimsHW, Dims3, DimsCHW, Dims4, DimsNCHW。不过 DimsCHW 和 DimsNCHW 将被 Dims3 和 Dims4 替代。

## LayerType
Tensorrt 支持的算子列表

## ITensor
Network 中 Tensor 的定义。包含了 Name, Dims, Type 等信息。比如 Network 中 Input Tensor。

## ILayer
网络模型节点的基类定义。

## PaddingMode
kEXPLICIT_ROUND_DOWN = 0, //!< Use explicit padding, rounding output size down.  
kEXPLICIT_ROUND_UP = 1,   //!< Use explicit padding, rounding output size up.  
kSAME_UPPER = 2,          //!< Use SAME padding, with prePadding <= postPadding.  
kSAME_LOWER = 3,          //!< Use SAME padding, with prePadding >= postPadding.  
kCAFFE_ROUND_DOWN = 4,    //!< Use CAFFE padding, rounding output size down, uses prePadding value.  
kCAFFE_ROUND_UP = 5       //!< Use CAFFE padding, rounding output size up, uses prePadding value.

## 各 Layer 的定义
继承 ILayer。

## INetworkDefinition
Network 结构的定义。提供了基本的接口，如 addInput, addConvolution, markOutput ...。将所有layer以及相应输入输出信息添加到 Network 中，即得到整个 Network。

## Calibrator
校准相关。CalibrationAlgoType, IInt8Calibrator, IInt8EntropyCalibrator, ...

## Builder
IBuilder 是为了根据 Network 创建 engine，而 engine 就是推理引擎。
### BuilderFlag
kFP16 = 0,         Enable FP16 layer selection, with FP32 fallback.   
kINT8 = 1,         Enable Int8 layer selection, with FP32 fallback with FP16 fallback if kFP16 also specified.  
kDEBUG = 2,        Enable debugging of layers via synchronizing after every layer.  
kGPU_FALLBACK = 3, Enable layers marked to execute on GPU if layer cannot execute on DLA.  
kSTRICT_TYPES = 4, Enables strict type constraints.  
kREFIT = 5,        Enable building a refittable engine.

如果不设置相关参数，默认使用 fp32 的模式进行推理。

### BuilderConfig
目前，config 的项大多数用默认值居多。需要根据实际情况设置的参数有
setMaxWorkspaceSize

### NetworkDefinitionCreationFlags
typedef uint32_t NetworkDefinitionCreationFlags;

enum class NetworkDefinitionCreationFlag : int
{
    kEXPLICIT_BATCH = 0,
    kEXPLICIT_PRECISION = 1,
};
可以通过 OR 来组合多个 flag 生成 NetworkDefinitionCreationFlags。如 1U << <uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)，1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH) | 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION)。

在 tensorrt 中有很多处以这样的方式将多个 flag 生成 flags，比如 BuilderFlag 和 BuilderFlags。

### IBuilder
createNetwork() 返回一个 INetworkDefinition 的类，这个创建的 Network 都是 implicit batch 的，不支持动态 shape，也就是在创建好 Network 之后进行推理时，需要指定 batch 的大小。

setMaxBatchSize(int) 在执行时支持的最大 batch_size，也是 tensorrt engine 优化的 batch_size。

buildEngineWithConfig(network, config) 根据 network 和 config 创建 engine。

createNetworkV2(flags) 根据 NetworkDefinitionCreationFlags 创建 Network，然后再向 Network 中添加各个 layer。

virtual addOptimizationProfile(const IOptimizationProfile* profile) 如果是动态 shape，即 createNetworkV2 的方式创建的 Network，这个函数至少需要被调用一次。添加一个 Profile 的过程如下，  
`network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)`  
`out = network.addInput(...)`  
`auto profile = builder->createOptimizationProfile();`  
`profile->setDimensions(input->getName(),` 
`OptProfileSelector::kMIN, Dims4{1, 1, 1, 1});`  
`profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 1, 28, 28});`  
`profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 1, 56, 56});`  
`config->addOptimizationProfile(profile);`  
`engine = builder->buildEngineWithConfig(*network, *config);`  
这个 Profile 允许 shape 在 [(1, 1, 1, 1), (1, 1, 56, 56)] 范围内的数据进行预测，但是 tensorrt 会根据 (1, 1, 28, 28) 这个 shape 进行优化。注意最后的 engine 是根据 network 和 config 共同创建。

### CreateBuilder
createInferBuilder(ILogger& logger)
