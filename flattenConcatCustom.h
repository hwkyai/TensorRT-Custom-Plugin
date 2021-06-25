/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_FlattenConcatCustom_PLUGIN_H
#define TRT_FlattenConcatCustom_PLUGIN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

#define LOG_ERROR(status)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace nvinfer1
{
namespace plugin
{
class CtcBeamSearchDecoderCustom : public IPluginV2Ext
{
public:
    CtcBeamSearchDecoderCustom(const void* data, size_t length);

    ~CtcBeamSearchDecoderCustom() override = default;

    CtcBeamSearchDecoderCustom() = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    size_t getWorkspaceSize(int) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize);

    bool supportsFormat(DataType type, PluginFormat format) const override;

    void detachFromContext() override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;
    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    const char* mPluginNamespace;
};

class FlattenConcatCustomPluginCreator : public BaseCreator
{
public:
    FlattenConcatCustomPluginCreator();

    ~FlattenConcatCustomPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_FlattenConcatCustom_PLUGIN_H
