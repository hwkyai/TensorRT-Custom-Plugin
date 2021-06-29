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
static const char* CTCBEAMSEARCHCUSTOM_PLUGIN_NAME{"CtcBeamSearchCustom_TRT"};

PluginFieldCollection CtcBeamSearchCustomPluginCreator::mFC{};
std::vector<PluginField> CtcBeamSearchCustomPluginCreator::mPluginAttributes;

CtcBeamSearchCustom::CtcBeamSearchCustom(const void* data, size_t length)
{
}

CtcBeamSearchCustom::~CtcBeamSearchCustom() {}

int CtcBeamSearchCustom::getNbOutputs() const
{
    return 1;
}

Dims CtcBeamSearchCustom::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    return Dims3(1,1,1);
}

int CtcBeamSearchCustom::initialize()
{
    return STATUS_SUCCESS;
}

void CtcBeamSearchCustom::terminate() {}

size_t CtcBeamSearchCustom::getWorkspaceSize(int) const
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
    // TODO
    return 1;
}

void CtcBeamSearchCustom::serialize(void* buffer) const
{
    // TODO: serialize
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
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CtcBeamSearchCustom::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
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
    ASSERT(index < 3);
    return DataType::kFLOAT;
}

void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
{
    // TODO: configure plugin
}

bool CtcBeamSearchCustom::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    return true;
    // return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
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

CtcBeamSearchCustomPluginCreator::CtcBeamSearchCustomPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("dummy", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
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
    return &mFC;
}

IPluginV2Ext* CtcBeamSearchCustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto* plugin = new CtcBeamSearchCustom();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* CtcBeamSearchCustomPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2Ext* plugin = new CtcBeamSearchCustom(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
