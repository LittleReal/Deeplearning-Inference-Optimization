#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace std;

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

// refer to https://github.com/onnx/onnx-tensorrt/blob/84b5be1d6fc03564f2c0dba85a2ee75bad242c2e/main.cpp
int main()
{
    Logger logger;
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto builder = nvinfer1::createInferBuilder(logger);  
    auto network = builder->createNetworkV2(explicitBatch);
    auto onnxparser  = nvonnxparser::createParser(*network, logger);

    string onnxfilename = "model.onnx";
    std::ifstream onnxfile(onnx_filename.c_str(),
                            std::ios::binary | std::ios::ate);
    std::streamsize filesize = onnxfile.tellg();
    onnxfile.seekg(0, std::ios::beg);
    std::vector<char> onnxbuf(file_size);
    if( !onnxfile.read(onnxbuf.data(), onnxbuf.size()) ) {
        cerr << "ERROR: Failed to read from file " << onnxfilename << endl;
        return -4;
    }

    auto success = onnxparser->parse(onnxbuf.data(), onnxbuf.size());
    if (!success) return -1;

    size_t max_batch_size = 1;
    size_t max_workspace_size = 1<<30;
    builder->setMaxBatchSize(max_batch_size);
    builder->setMaxWorkspaceSize(max_workspace_size);

    auto engine = builder->buildCudaEngine(*network.get());
    auto plan = engine->serialize();
    engine_filename = "model.trt"
    ofstream engine_file(engine_filename.c_str());
    if (!engine_file) {
      cerr << "Failed to open output file for writing: "
           << engine_filename << endl;
      return -1;
    }

    engine_file.write((char*)plan->data(), plan->size());
    engine_file.close();

    return 0;
}