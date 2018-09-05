//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <string>

#include <mkldnn.hpp>

#include "mkldnn_invoke.hpp"
#include "ngraph/runtime/ccpu/ccpu_runtime_context.hpp"
#include "ngraph/runtime/ccpu/mkldnn_utils.hpp"

mkldnn::engine ngraph::runtime::ccpu::mkldnn_utils::global_cpu_engine(mkldnn::engine::cpu, 0);

extern "C" void ngraph::runtime::ccpu::mkldnn_utils::set_memory_ptr(CCPURuntimeContext* ctx,
                                                                    size_t primitive_index,
                                                                    void* ptr)
{
    auto primitive = static_cast<mkldnn::memory*>(ctx->mkldnn_primitives[primitive_index]);
    primitive->set_data_handle(ptr);
}

extern "C" void
    ngraph::runtime::ccpu::mkldnn_utils::mkldnn_invoke_primitive(CCPURuntimeContext* ctx,
                                                                 size_t primitive_index)
{
    mkldnn::stream s(mkldnn::stream::kind::eager);
    try
    {
        s.submit({*ctx->mkldnn_primitives[primitive_index]}).wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + e.message);
    }
}