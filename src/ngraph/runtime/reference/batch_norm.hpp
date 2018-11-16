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

#include <cmath>
#include <iostream>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void batch_norm_training_with_intermediates(double eps,
                                                        const T* gamma,
                                                        const T* beta,
                                                        const T* input,
                                                        T* normed_input,
                                                        T* mean,
                                                        T* variance,
                                                        T* centered_input,
                                                        T* scaled_input,
                                                        const Shape& input_shape)
            {
                std::cerr << "REFERENCE!!" << std::endl;

                auto eps_casted = static_cast<T>(eps);
                auto channels = input_shape[1];

                // We use these objects to iterate over the indices in a channel.
                // The start and end points for the channel axis are modified in the loop.
                Coordinate start_corner;
                Coordinate end_corner;
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    start_corner.push_back(0);
                    end_corner.push_back(input_shape[i]);
                }

                for (size_t c = 0; c < channels; c++)
                {
                    T channel_sum = 0;

                    start_corner[1] = c;
                    end_corner[1] = c + 1;

                    // Compute the mean
                    CoordinateTransform input_transform(input_shape, start_corner, end_corner);
                    for (Coordinate input_coord : input_transform)
                    {
                        channel_sum += input[input_transform.index(input_coord)];
                    }
                    T channel_mean = channel_sum / (shape_size(input_shape) / channels);
                    mean[c] = channel_mean;
                    T channel_diff_square_sum = 0;
                    for (Coordinate input_coord : input_transform)
                    {
                        auto centered = input[input_transform.index(input_coord)] - channel_mean;
                        centered_input[input_transform.index(input_coord)] = centered;
                        channel_diff_square_sum += centered * centered;
                    }
                    T channel_var = channel_diff_square_sum / (shape_size(input_shape) / channels);
                    variance[c] = channel_var;

                    auto channel_gamma = gamma[c];
                    auto channel_beta = beta[c];
                    T scale = channel_gamma / std::sqrt(channel_var + eps_casted);

                    // Compute the normalized output
                    for (Coordinate input_coord : input_transform)
                    {
                        auto input_index = input_transform.index(input_coord);
                        scaled_input[input_index] = centered_input[input_index] * scale;
                        normed_input[input_index] = scaled_input[input_index] + channel_beta;
                    }
                }
            }

            template <typename T>
            void batch_norm_training(double eps,
                                     const T* gamma,
                                     const T* beta,
                                     const T* input,
                                     T* normed_input,
                                     T* mean,
                                     T* variance,
                                     const Shape& input_shape)
            {
                std::vector<T> centered(shape_size(input_shape));
                std::vector<T> normalized(shape_size(input_shape));
                batch_norm_training_with_intermediates(eps,
                                                       gamma,
                                                       beta,
                                                       input,
                                                       normed_input,
                                                       mean,
                                                       variance,
                                                       centered.data(),
                                                       normalized.data(),
                                                       input_shape);
            }

            template <typename T>
            void batch_norm_inference(double eps,
                                      const T* gamma,
                                      const T* beta,
                                      const T* input,
                                      const T* mean,
                                      const T* variance,
                                      T* normed_input,
                                      const Shape& input_shape)
            {
                auto eps_casted = static_cast<T>(eps);
                CoordinateTransform input_transform(input_shape);

                for (Coordinate input_coord : input_transform)
                {
                    auto channel_num = input_coord[1];
                    auto channel_gamma = gamma[channel_num];
                    auto channel_beta = beta[channel_num];
                    auto channel_mean = mean[channel_num];
                    auto channel_var = variance[channel_num];

                    auto input_index = input_transform.index(input_coord);
                    auto normalized =
                        (input[input_index] - channel_mean) / (std::sqrt(channel_var + eps_casted));
                    normed_input[input_index] = normalized * channel_gamma + channel_beta;
                }
            }

            template <typename T>
            void batch_norm_backprop(double eps,
                                     const T* gamma,
                                     const T* beta,
                                     const T* input,
                                     const T* mean,
                                     const T* variance,
                                     const T* delta,
                                     T* delta_input,
                                     T* delta_gamma,
                                     T* delta_beta,
                                     const Shape& input_shape)
            {
                size_t channel_axis = 1;
                auto eps_casted = static_cast<T>(eps);
                auto num_channels = input_shape[channel_axis];
                Shape moment_shape = Shape{num_channels};
                auto input_num_elements = shape_size(input_shape);
                auto elements_per_channel = input_num_elements / num_channels;

                Coordinate start_corner;
                Coordinate end_corner;
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    start_corner.push_back(0);
                    end_corner.push_back(input_shape[i]);
                }
                // The forward computation in gory detail
                // input[., C, ...]
                // gamma[C]
                // beta[C]
                // mu_sum[c:C] = sum(input[., c, ...])
                // mu[c:C] = mu_sum[c]/element_per_channel
                // centered_input[., c:C, ...] = input[., c, ...] - mu[c]
                // square[., c:C, ...] = centered_input[., c, ...]^2
                // var_sum[c:C] = sum(square[., c, ...])
                // var[c:C] = var_sum[c]/elements_per_channel
                // var_eps[c:C] = var[c:C]+epsilon
                // inv_sqrt[c:C] = 1/sqrt(var_eps[c])
                // gammad[c:C] = gamma[c]*inv_sqrt[c]
                // scaled[., c:C, ...] = centered_input[., c, ...]*gammad[c]
                // betad[., c:C, ...] = scaled[., c, ...]+beta[c]

                for (auto c = 0; c < num_channels; ++c)
                {
                    start_corner[channel_axis] = c;
                    end_corner[channel_axis] = c + 1;

                    CoordinateTransform input_transform(input_shape, start_corner, end_corner);
                    T beta_sum = 0;
                    T var = variance[c];
                    T mu = mean[c];
                    T one = 1;
                    T inv_sqrt_var_eps = one / std::sqrt(var + eps);
                    T gammad_delta = 0;
                    for (Coordinate input_coord : input_transform)
                    {
                        auto idx = input_transform.index(input_coord);
                        delta_input[idx] = 0;
                        auto delta_idx = delta[idx];
                        beta_sum += delta_idx;

                        auto input_idx = input[idx];
                        auto centered = input_idx - mu;
                        gammad_delta += centered * delta_idx;
                    }
                    delta_gamma[c] = gammad_delta * inv_sqrt_var_eps;
                    delta_beta[c] = beta_sum;
                }
            }
        }
    }
}
