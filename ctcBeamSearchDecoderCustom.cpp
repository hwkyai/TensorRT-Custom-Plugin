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
    const char* d = static_cast<const char*>(data);
    const char* const a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    ASSERT(mConcatAxisID >= 1 && mConcatAxisID <= 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);

    mInputConcatAxis.resize(mNumInputs);
    std::for_each(mInputConcatAxis.begin(), mInputConcatAxis.end(), [&](int& inp) { inp = read<int>(d); });

    mCHW = read<nvinfer1::DimsCHW>(d);

    mCopySize.resize(mNumInputs);
    std::for_each(mCopySize.begin(), mCopySize.end(), [&](size_t& inp) { inp = read<size_t>(d); });

    ASSERT(d == a + length);
}

CtcBeamSearchCustom::~CtcBeamSearchCustom() {}

int CtcBeamSearchCustom::getNbOutputs() const
{
    return 1;
}

Dims CtcBeamSearchCustom::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims >= 1);
    ASSERT(index == 0);

    mNumInputs = nbInputDims;
    mCopySize.resize(mNumInputs);
    mInputConcatAxis.resize(mNumInputs);
    int outputConcatAxis = 0;

    for (int i = 0; i < nbInputDims; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputs[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            ASSERT(inputs[i].d[0] == inputs[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputs[i].d[1] == inputs[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputs[i].d[2] == inputs[0].d[2]);
        }
        flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        outputConcatAxis += flattenInput;
    }

    return DimsCHW(mConcatAxisID == 1 ? outputConcatAxis : 1, mConcatAxisID == 2 ? outputConcatAxis : 1,
        mConcatAxisID == 3 ? outputConcatAxis : 1);
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
    ASSERT(mConcatAxisID != 0);
    // mCHW is the first input tensor
    int numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());

    // Num concats will be proportional to number of samples in a batch
    if (!mIgnoreBatch)
    {
        numConcats *= batchSize;
    }

    auto* output = static_cast<float*>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i)
    {
        const auto* input = static_cast<const float*>(inputs[i]);
        for (int n = 0; n < numConcats; ++n)
        {
            CUBLASASSERT(cublasScopy(mCublas, mInputConcatAxis[i], input + n * mInputConcatAxis[i], 1,
                output + (n * mOutputConcatAxis + offset), 1));
        }
        offset += mInputConcatAxis[i];
    }

    return STATUS_SUCCESS;
}

size_t CtcBeamSearchCustom::getSerializationSize() const
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims)
        + (sizeof(decltype(mCopySize)::value_type) * mNumInputs);
}

void CtcBeamSearchCustom::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* const a = d;
    write(d, mIgnoreBatch);
    write(d, mConcatAxisID);
    write(d, mOutputConcatAxis);
    write(d, mNumInputs);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mInputConcatAxis[i]);
    }
    write(d, mCHW);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mCopySize[i]);
    }
    ASSERT(d == a + getSerializationSize());
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CtcBeamSearchCustom::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
    mCublas = cublasContext;
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
    return "CtcBeamSearchCustom_TRT";
}

const char* CtcBeamSearchCustom::getPluginVersion() const
{
    return "1";
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
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

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
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "axis"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mConcatAxisID = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "ignoreBatch"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
        }
    }

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
