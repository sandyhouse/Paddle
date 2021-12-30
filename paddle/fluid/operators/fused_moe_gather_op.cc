#include "paddle/fluid/operators/fused_moe_gather_op.h"

namespace paddle{
namespace operators{

class FusedMOEGatherOp : public framework::OperatorWithKernel{
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of FusedMOEGatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Pos"), true,
                      platform::errors::InvalidArgument(
                          "Input(Pos) of FusedMOEGatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("local_count"), true,
                      platform::errors::InvalidArgument(
                          "Input(local_count) of FusedMOEGatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("global_count"), true,
                      platform::errors::InvalidArgument(
                          "Input(global_count) of FusedMOEGatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of FusedMOEGatherOp should not be null."));
    int ring_id = ctx->Attrs().Get<int>("ring_id");

    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global scatter op must be non-negative.",
            ring_id));
    
    auto input_dims = ctx->GetInputDim("X");
    auto pos_dim = ctx->GetInputDim("Pos");

    // dim check
    PADDLE_ENFORCE_EQ(input_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The input tensor's dimension must be 2. "
                          "But received input's dimension = %d.",
                          input_dims.size()));
    PADDLE_ENFORCE_EQ(
        pos_dim.size() == 1 || (pos_dim.size() == 2 && pos_dim[1] == 1),
        true, platform::errors::InvalidArgument(
                  "The 'shape' of Input(Pos) must be 1-D tensor. "
                  "But received: the 'shape' of Input(Pos) is [%s], "
                  "the dimension of Input(Index) is [%d].",
                  pos_dim, pos_dim.size()));

    PADDLE_ENFORCE_EQ(pos_dim[0] != 0, true,
                      platform::errors::InvalidArgument(
                          "The length of Input(Pos) can't be 0."));

    framework::DDim out_dims = framework::make_ddim({-1, -1});
    ctx->SetOutputDim("Out", out_dims);

}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }

};

class FusedMOEGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor gather.");
    AddInput("Pos", "the 1-D tensor containing the indices to index.");
    AddInput("local_count",
             "(Tensor) Tensor which has n_expert * world_size elements that "
             "indicates"
             "how many data needed to be sent to each expert.");
    AddInput("global_count",
             "(Tensor) Tensor which has n_expert * world_size elements that "
             "indicates"
             "how many data needed to be received from each expert.");

    AddAttr<int>("out_batch_size", "(int),the shape[0] of raw input.");
    AddAttr<int>("world_size", "(int),the size of expert group.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("dim", "the dimension in which we index.").SetDefault(0);
    //AddAttr<bool>(
    //    "use_calc_stream",
    //    "(bool default false) eject CUDA operations to calculation stream.")
    //    .SetDefault(false);
    AddOutput("Out", "(Tensor) the result of fused_moe_gather.");
    AddComment(R"DOC(
Fuse MoE Gather Operator
fuse the global_gather and local_gather in MoE.
)DOC");
  }
};


class FusedMOEGatherOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) should not be null"));
    
    PADDLE_ENFORCE_EQ(ctx->HasInput("Pos"), true,
                      platform::errors::InvalidArgument(
                          "Input(Pos) should not be null."));
    
    PADDLE_ENFORCE_EQ(ctx->HasInput("local_count"), true,
                      platform::errors::InvalidArgument(
                          "Input(local_count) should not be null."));
    
    PADDLE_ENFORCE_EQ(ctx->HasInput("global_count"), true,
                      platform::errors::InvalidArgument(
                          "Input(global_count) should not be null."));

    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) should not be null."));
    
    framework::DDim out_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);

  }

  framework::OpKernelType GetExpectedKernelType(const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedMOEGatherOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("fused_moe_gather_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("Pos", this->Input("Pos"));
    retv->SetInput("local_count", this->Input("local_count"));
    retv->SetInput("global_count", this->Input("global_count"));
    // just to use the shape of X. register no need buffer.
    retv->SetInput("X", this->Input("X"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(fused_moe_gather, ops::FusedMOEGatherOp,
                  ops::FusedMOEGatherOpMaker,
                  ops::FusedMOEGatherOpGradMaker<paddle::framework::OpDesc>,
                  ops::FusedMOEGatherOpGradMaker<paddle::imperative::OpBase>)

REGISTER_OPERATOR(fused_moe_gather_grad,
                  ops::FusedMOEGatherOpGrad);

REGISTER_OP_CPU_KERNEL(fused_moe_gather, ops::FusedMOEGatherOpCPUKernel<paddle::platform::CPUDeviceContext, float>,
                       ops::FusedMOEGatherOpCPUKernel<paddle::platform::CPUDeviceContext, double>,
                       ops::FusedMOEGatherOpCPUKernel<paddle::platform::CPUDeviceContext, int>,
                       ops::FusedMOEGatherOpCPUKernel<paddle::platform::CPUDeviceContext, int64_t>);

