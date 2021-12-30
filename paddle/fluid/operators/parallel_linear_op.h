/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename RepeatsT = int>
void RepeatsTensor2IndexTensor(const LoDTensor& repeats, LoDTensor* index, const platform::DeviceContext& ctx) {
  LoDTensor repeats_cpu_copy;
  if (!platform::is_cpu_place(repeats.place())) {
    framework::TensorCopySync(repeats, platform::CPUPlace(), &repeats_cpu_copy);
  }
  const RepeatsT* repeats_data = platform::is_cpu_place(repeats.place())
                                     ? repeats.data<RepeatsT>()
                                     : repeats_cpu_copy.data<RepeatsT>();

  int64_t index_size = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    index_size += repeats_data[i];
  }
  std::vector<RepeatsT> index_vec(index_size);
  int offset = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    std::fill_n(index_vec.begin() + offset, repeats_data[i], i);
    offset += repeats_data[i];
  }
  index->Resize(framework::make_ddim({index_size}));

  paddle::framework::TensorFromVector<RepeatsT>(index_vec, ctx, index);
}

template <typename DeviceContext, typename T>
class ParallelLinearOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::Unimplemented("BatchFC only supports GPU now."));
  }
};
}  // namespace operators
}  // namespace paddle
