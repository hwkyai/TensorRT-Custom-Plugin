/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "ctcBeamSearchDecoderCustom.h"
#include <algorithm>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::CtcBeamSearchCustom;
using nvinfer1::plugin::CtcBeamSearchCustomPluginCreator;

static const char* CTCBEAMSEARCHCUSTOM_PLUGIN_VERSION{"1"};
static const char* CTCBEAMSEARCHCUSTOM_PLUGIN_NAME{"CTCBeamSearchDecoder"};

PluginFieldCollection CtcBeamSearchCustomPluginCreator::mFC = {};
std::vector<PluginField> CtcBeamSearchCustomPluginCreator::mPluginAttributes;

CtcBeamSearchCustom::~CtcBeamSearchCustom() {}

int CtcBeamSearchCustom::getNbOutputs() const
{
    return 1;
}

Dims CtcBeamSearchCustom::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(index < nbInputDims);
    return inputs[index];
}

int CtcBeamSearchCustom::initialize()
{
    return STATUS_SUCCESS;
}

void CtcBeamSearchCustom::terminate() {}

size_t CtcBeamSearchCustom::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int CtcBeamSearchCustom::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    auto* output = static_cast<float*>(outputs[0]);
    auto* input = static_cast<const float*>(inputs[0]);

    *output = *input;

    return STATUS_SUCCESS;
}

size_t CtcBeamSearchCustom::getSerializationSize() const
{
    return 0;
}

void CtcBeamSearchCustom::serialize(void* buffer) const
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CtcBeamSearchCustom::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void CtcBeamSearchCustom::detachFromContext() {}

// Return true if output tensor is broadcast across a batch.
bool CtcBeamSearchCustom::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    ASSERT(outputIndex < nbInputs)
    return inputIsBroadcasted[outputIndex];
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CtcBeamSearchCustom::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return true;
}

// Set plugin namespace
void CtcBeamSearchCustom::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* CtcBeamSearchCustom::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType CtcBeamSearchCustom::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index < nbInputs);
    return inputTypes[index];
}

void CtcBeamSearchCustom::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
{
}

bool CtcBeamSearchCustom::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    return true;
}
const char* CtcBeamSearchCustom::getPluginType() const
{
    return CTCBEAMSEARCHCUSTOM_PLUGIN_NAME;
}

const char* CtcBeamSearchCustom::getPluginVersion() const
{
    return CTCBEAMSEARCHCUSTOM_PLUGIN_VERSION;
}

void CtcBeamSearchCustom::destroy()
{
    delete this;
}

IPluginV2IOExt* CtcBeamSearchCustom::clone() const
{
    auto* plugin = new CtcBeamSearchCustom();
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

const char* CtcBeamSearchCustomPluginCreator::getPluginName() const
{
    return CTCBEAMSEARCHCUSTOM_PLUGIN_NAME;
}

const char* CtcBeamSearchCustomPluginCreator::getPluginVersion() const
{
    return CTCBEAMSEARCHCUSTOM_PLUGIN_VERSION;
}

const PluginFieldCollection* CtcBeamSearchCustomPluginCreator::getFieldNames()
{
    return nullptr;
}

CtcBeamSearchCustom* CtcBeamSearchCustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto* plugin = new CtcBeamSearchCustom();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

CtcBeamSearchCustom* CtcBeamSearchCustomPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    // IPluginV2Ext* plugin = new CtcBeamSearchCustom();
    auto* plugin = new CtcBeamSearchCustom();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
