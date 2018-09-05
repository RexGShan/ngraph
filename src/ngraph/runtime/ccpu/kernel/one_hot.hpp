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

#pragma once

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/runtime/ccpu/kernel/eigen_thread_pool.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ccpu
        {
            namespace kernel
            {
                template <typename ElementType>
                void one_hot_rank_0(void* arg,
                                    void* out,
                                    const Shape& out_shape,
                                    size_t one_hot_axis)

                {
                    Eigen::array<Eigen::Index, 1> out_dims;
                    out_dims[0] = out_shape[0];

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> out_tensor(
                        static_cast<ElementType*>(out), out_dims);

                    out_tensor.setZero();
                    auto pos_raw = (static_cast<ElementType*>(arg))[0];
                    size_t pos = pos_raw;
                    out_tensor(pos) = 1;
                }

                template <typename ElementType>
                void one_hot_rank_1(void* arg,
                                    void* out,
                                    const Shape& arg_shape,
                                    const Shape& out_shape,
                                    const Strides& out_strides,
                                    size_t one_hot_axis)

                {
                    Eigen::array<Eigen::Index, 2> out_dims;
                    Eigen::array<Eigen::Index, 1> in_dims;
                    out_dims[0] = out_shape[0];
                    out_dims[1] = out_shape[1];
                    in_dims[0] = arg_shape[0];

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 2, Eigen::RowMajor>> out_tensor(
                        static_cast<ElementType*>(out), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in_tensor(
                        static_cast<ElementType*>(arg), in_dims);

                    out_tensor.setZero();
                    for (size_t i = 0; i < arg_shape[0]; i++)
                    {
                        auto pos_raw = in_tensor(i);
                        size_t pos = pos_raw;
                        one_hot_axis == 0 ? out_tensor(pos, i) = 1 : out_tensor(i, pos) = 1;
                    }
                }

                template <typename ElementType>
                void one_hot_rank_2_or_more(void* arg,
                                            void* out,
                                            const Shape& arg_shape,
                                            const Shape& out_shape,
                                            size_t one_hot_axis)

                {
                    reference::one_hot<ElementType>(static_cast<const ElementType*>(arg),
                                                    static_cast<ElementType*>(out),
                                                    arg_shape,
                                                    out_shape,
                                                    one_hot_axis);
                }
            }
        }
    }
}