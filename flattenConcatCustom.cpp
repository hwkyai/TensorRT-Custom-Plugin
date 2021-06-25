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
#include "flattenConcatCustom.h"
#include <algorithm>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::CtcBeamSearchDecoderCustom;
using nvinfer1::plugin::FlattenConcatCustomPluginCreator;

static const char* FlattenConcatCustom_PLUGIN_VERSION{"1"};
static const char* FlattenConcatCustom_PLUGIN_NAME{"CtcBeamSearchDecoderCustom"};

CtcBeamSearchDecoderCustom::CtcBeamSearchDecoderCustom(const void* data, size_t length)
{
    ASSERT(getSerializationSize() == 0);
}

int CtcBeamSearchDecoderCustom::getNbOutputs() const
{
    return -1;
}

Dims CtcBeamSearchDecoderCustom::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    
}

int CtcBeamSearchDecoderCustom::initialize()
{
    return STATUS_SUCCESS;
}

size_t CtcBeamSearchDecoderCustom::getWorkspaceSize(int) const
{
    return 0;
}

int CtcBeamSearchDecoderCustom::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    &outputs = inputs;
    return 0;
}

size_t CtcBeamSearchDecoderCustom::getSerializationSize() const
{
    return 0;
}

void CtcBeamSearchDecoderCustom::serialize(void* buffer) const
{
    ASSERT(getSerializationSize() == 0);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CtcBeamSearchDecoderCustom::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void CtcBeamSearchDecoderCustom::detachFromContext() {}

// Return true if output tensor is broadcast across a batch.
bool CtcBeamSearchDecoderCustom::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CtcBeamSearchDecoderCustom::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Set plugin namespace
void CtcBeamSearchDecoderCustom::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* CtcBeamSearchDecoderCustom::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType CtcBeamSearchDecoderCustom::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

bool CtcBeamSearchDecoderCustom::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}
const char* CtcBeamSearchDecoderCustom::getPluginType() const
{
    return "CtcBeamSearchDecoderCustom";
}

const char* CtcBeamSearchDecoderCustom::getPluginVersion() const
{
    return "1";
}

void CtcBeamSearchDecoderCustom::destroy()
{
    delete this;
}

IPluginV2Ext* CtcBeamSearchDecoderCustom::clone() const
{
    auto* plugin = new CtcBeamSearchDecoderCustom();
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

FlattenConcatCustomPluginCreator::FlattenConcatCustomPluginCreator()
{
}

const char* FlattenConcatCustomPluginCreator::getPluginName() const
{
    return FlattenConcatCustom_PLUGIN_NAME;
}

const char* FlattenConcatCustomPluginCreator::getPluginVersion() const
{
    return FlattenConcatCustom_PLUGIN_VERSION;
}

IPluginV2Ext* FlattenConcatCustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto* plugin = new CtcBeamSearchDecoderCustom();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* FlattenConcatCustomPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2Ext* plugin = new CtcBeamSearchDecoderCustom(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
REGISTER_TENSORRT_PLUGIN(FlattenConcatCustomPluginCreator);


