#pragma once
#include "paddle/fluid/framework/op_registry.h"
#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedMOEScatterOpCPUKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unavailable(
        "Do not support the cpu kernel of fuse MOE scatter op now."));
  }
};

}  // namespace operators
}  // namespace paddle
