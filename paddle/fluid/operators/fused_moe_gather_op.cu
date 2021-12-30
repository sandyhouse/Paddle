#include <algorithm>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/scatter.cu.h"
#include "paddle/fluid/operators/fused_moe_gather_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"

template <typename T, typename IndexT>
__global__ void index_select_cuda_kernel(const T* input, T* output,
                                         const IndexT* index, int64_t N,
                                         int64_t stride, int64_t size,
                                         int64_t delta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t pre_idx = idx / (stride * size);
  int64_t dim_idx = idx % (stride * size) / stride;
  IndexT src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
  output[idx] = input[input_idx];
}

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class FusedMOEGatherOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override { 
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto* index = ctx.Input<LoDTensor>("Pos");
    int world_size = ctx.Attr<int>("world_size");
    int out_batch_size = ctx.Attr<int>("out_batch_size");

    auto* out = ctx.Output<LoDTensor>("Out");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    
    Tensor local_output_buf;
    local_output_buf = ctx.AllocateTmpTensor<T, DeviceContext>(
        {in->dims()}, dev_ctx);
    // global gather
    if(world_size > 1){
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto* local_count = ctx.Input<framework::LoDTensor>("local_count");
    auto* global_count = ctx.Input<framework::LoDTensor>("global_count");
    auto local_count_type = local_count->type();
    auto global_count_type = global_count->type();
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    auto local_count_len = 0;

    framework::Tensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
      local_count_len = local_count->numel();
    } else {
      framework::TensorCopySync(*local_count, platform::CPUPlace(),
                                &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
      local_count_len = cpu_local_count.numel();
    }

    framework::Tensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(*global_count, platform::CPUPlace(),
                                &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
    }

    ncclDataType_t dtype = platform::ToNCCLDataType(in->type());

    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global gather op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    cudaStream_t stream = nullptr;
    // default use_calc_stream
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    int nranks = comm->nranks();
    auto in_feat = in->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;

    auto fwd_count = 0;
    //framework::Tensor count;
    //framework::DDim fwd_count_dims = framework::make_ddim({1});
    //std::vector<int> out_reduce_dims;
    //out_reduce_dims.push_back(0);
    //auto fwd_data_tmp = count.mutable_data<T>(fwd_count_dims, place);
    //TensorReduceFunctorImpl<int64_t, int64_t, CustomSum>(
    //    *local_count, &count, out_reduce_dims, stream);
    //framework::Tensor count_tensor;
    //framework::TensorCopy(count, platform::CPUPlace(), &count_tensor);
    //auto fwd_count = count_tensor.data<int64_t>()[0];

    for (auto i = 0; i < local_count_len; ++i) {
      fwd_count += cpu_local_count_data[i];
    }
    framework::DDim out_dims = framework::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }
    auto send_ptr = 0;
    auto send_buf = in->data<T>();
    auto recv_buf = local_output_buf.mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclSend(send_buf + send_ptr * in_feat,
                                          cpu_global_count_data[idx] * in_feat,
                                          dtype, j, comm->comm(), stream));
          send_ptr += cpu_global_count_data[idx];
        }
        if (cpu_local_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclRecv(recv_buf + expert_ptr[idx] * in_feat,
                                          cpu_local_count_data[idx] * in_feat,
                                          dtype, j, comm->comm(), stream));
        }
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    }
#else
    PADDLE_THROW(
        platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif 

    }
    else{
	  local_output_buf.ShareDataWith(*in);
    }

    //local gather
    //Tensor inp_buf;
    //inp_buf = ctx.AllocateTmpTensor<T, DeviceContext>(
    //                      {out_batch_size, local_output_buf.dims()[1]}, dev_ctx);
    //inp_buf.mutable_data<T>(ctx.GetPlace()); 
    //auto &eigen_place = *ctx.template device_context<platform::CUDADeviceContext>()
    //                          .eigen_device();
    //auto inp_buf_eigen = framework::EigenVector<T>::Flatten(inp_buf);
    //inp_buf_eigen.device(eigen_place) = inp_buf_eigen.constant(static_cast<T>(0));
    
   
    out->mutable_data<T>({out_batch_size, local_output_buf.dims()[1]}, ctx.GetPlace()); 
    auto &eigen_place = *ctx.template device_context<platform::CUDADeviceContext>()
                              .eigen_device();
    auto inp_buf_eigen = framework::EigenVector<T>::Flatten(*out);
    inp_buf_eigen.device(eigen_place) = inp_buf_eigen.constant(static_cast<T>(0));
    if (index->dims().size() == 1 && index->dims()[0] == 0){
        // back 0
        return ;
    }else{
        //framework::TensorCopy(inp_buf, ctx.GetPlace(), out);
        // use template class to support int32_t and int64_t
        const auto &index_type = index->type();
        bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
        PADDLE_ENFORCE_EQ(
            index_type_match, true,
            platform::errors::InvalidArgument(
                "scatter_op Index holds the wrong type, it holds [%s],"
                "but desires to be [%s] or [%s].",
                paddle::framework::DataTypeToString(index_type),
                paddle::framework::DataTypeToString(
                    framework::proto::VarType::INT32),
                paddle::framework::DataTypeToString(
                    framework::proto::VarType::INT64)));
        if (index_type == framework::proto::VarType::INT32) {
            GPUScatterAssign<T, int32_t>(ctx, local_output_buf, *index, out, true);
        } else {
            GPUScatterAssign<T, int64_t>(ctx, local_output_buf, *index, out, true);
        }

    }

  }
};


template <typename DeviceContext, typename T>
class FusedMOEGatherOpGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
       auto* output_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
       auto* in_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
       auto* index = ctx.Input<LoDTensor>("Pos");
       int world_size = ctx.Attr<int>("world_size");
       int out_batch_size = ctx.Attr<int>("out_batch_size");

       auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    
       Tensor grad_out_buf;
       if(index->dims().size() == 1 && index->dims()[0] == 0){
           grad_out_buf = ctx.AllocateTmpTensor<T, DeviceContext>(
                       {0, output_grad->dims()[1]}, dev_ctx);
           grad_out_buf.mutable_data<T>(ctx.GetPlace());
       }else{
           grad_out_buf = ctx.AllocateTmpTensor<T, DeviceContext>(
                       {index->dims()[0], output_grad->dims()[1]}, dev_ctx);
	   auto* local_output_buf_data = grad_out_buf.mutable_data<T>(ctx.GetPlace());
	   
	   int dim = ctx.Attr<int>("dim");
           auto* in_data = output_grad->data<T>();
           auto input_dim = output_grad->dims();
           auto local_output_buf_dim = grad_out_buf.dims();
           dim = dim >= 0 ? dim : dim + input_dim.size();
	   auto stride_dim = framework::stride(input_dim);
	   int64_t stride = stride_dim[dim];
	   int64_t size = local_output_buf_dim[dim];
	   int64_t delta = input_dim[dim] - size;
	   int64_t numel = grad_out_buf.numel();
	   auto index_type = index->type();
	   auto stream = ctx.template device_context<platform::CUDADeviceContext>().stream();
	   // local_scatter
	   if (index_type == framework::proto::VarType::INT64) {
		const int64_t* index_data = index->data<int64_t>();
		index_select_cuda_kernel<T, int64_t> <<<
		(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
		 PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_data, local_output_buf_data, index_data,
						       numel, stride, size, delta);
#ifdef PADDLE_WITH_HIP
	     PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(stream));
#else
	     PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#endif
	   } else {
	     const int* index_data = index->data<int>();
	     index_select_cuda_kernel<T, int><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_data, local_output_buf_data, index_data, numel, stride, size, delta);
#ifdef PADDLE_WITH_HIP
	     PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(stream));
#else
	     PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#endif
	   }
       }

       if (world_size <= 1){
	   in_grad->ShareDataWith(grad_out_buf);
       }else{
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703

	    auto* local_count = ctx.Input<framework::LoDTensor>("local_count");
	    auto* global_count = ctx.Input<framework::LoDTensor>("global_count");
	    auto local_count_type = local_count->type();
	    auto global_count_type = global_count->type();
	    if (local_count_type != framework::proto::VarType::INT64) {
	      PADDLE_THROW(platform::errors::InvalidArgument(
		  "Please use int64 type in local_count."));
	    }
	    if (global_count_type != framework::proto::VarType::INT64) {
	      PADDLE_THROW(platform::errors::InvalidArgument(
		  "Please use int64 type in global_count."));
	    }

	    const int64_t* cpu_local_count_data;
	    const int64_t* cpu_global_count_data;
	    framework::Tensor cpu_local_count;
	    if (platform::is_cpu_place(local_count->place())) {
	      cpu_local_count_data = local_count->data<int64_t>();
	    } else {
	      framework::TensorCopySync(*local_count, platform::CPUPlace(),
					&cpu_local_count);
	      cpu_local_count_data = cpu_local_count.data<int64_t>();
	    }

	    auto global_count_len = 0;
	    framework::Tensor cpu_global_count;
	    if (platform::is_cpu_place(global_count->place())) {
	      cpu_global_count_data = global_count->data<int64_t>();
	      global_count_len = global_count->numel();
	    } else {
	      framework::TensorCopySync(*global_count, platform::CPUPlace(),
					&cpu_global_count);
	      cpu_global_count_data = cpu_global_count.data<int64_t>();
	      global_count_len = cpu_global_count.numel();
	    }

	    ncclDataType_t dtype = platform::ToNCCLDataType(grad_out_buf.type());

	    int ring_id = ctx.Attr<int>("ring_id");
	    PADDLE_ENFORCE_GE(
		ring_id, 0,
		platform::errors::InvalidArgument(
		    "The ring_id (%d) for global scatter op must be non-negative.",
		    ring_id));

	    auto place = ctx.GetPlace();
	    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
	    cudaStream_t stream = nullptr;
            auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
	    stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
	    //if (ctx.Attr<bool>("use_calc_stream")) {
	    //  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
	    //  stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
	    //} else {
	    //  stream = comm->stream();
	    //}
	    int nranks = comm->nranks();
	    auto in_feat = grad_out_buf.dims()[1];
	    auto n_expert = local_count->dims()[0] / nranks;
	    int64_t fwd_count = 0;

	    for (auto i = 0; i < global_count_len; ++i) {
	      fwd_count += cpu_global_count_data[i];
	    }
	    framework::DDim out_dims = framework::make_ddim({fwd_count, in_feat});
	    int64_t* expert_ptr = new int64_t[n_expert * nranks];
	    expert_ptr[0] = 0;
	    auto tot_experts = n_expert * nranks;
        
	    for (auto i = 1; i < tot_experts; ++i) {
	      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
	    }
    
        

	    auto recv_ptr = 0;
	    auto send_buf = grad_out_buf.data<T>();
	    auto recv_buf = in_grad->mutable_data<T>(out_dims, place);

        
	    for (auto i = 0; i < n_expert; ++i) {
	      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
	      for (auto j = 0; j < nranks; ++j) {
		int idx = i + j * n_expert;
		if (cpu_local_count_data[idx]) {
		  PADDLE_ENFORCE_GPU_SUCCESS(
		      platform::dynload::ncclSend(send_buf + expert_ptr[idx] * in_feat,
						  cpu_local_count_data[idx] * in_feat,
						  dtype, j, comm->comm(), stream));
		}
		if (cpu_global_count_data[idx]) {
		  PADDLE_ENFORCE_GPU_SUCCESS(
		      platform::dynload::ncclRecv(recv_buf + recv_ptr * in_feat,
						  cpu_global_count_data[idx] * in_feat,
						  dtype, j, comm->comm(), stream));
		  recv_ptr += cpu_global_count_data[idx];
		}

	      }
	      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
	    }

#else
	    PADDLE_THROW(
		platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
	    PADDLE_THROW(
		platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
    }

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(fused_moe_gather, ops::FusedMOEGatherOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
                        ops::FusedMOEGatherOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
                        ops::FusedMOEGatherOpCUDAKernel<paddle::platform::CUDADeviceContext, int>,
                        ops::FusedMOEGatherOpCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>,
                        ops::FusedMOEGatherOpCUDAKernel<paddle::platform::CUDADeviceContext, plat::float16>);


REGISTER_OP_CUDA_KERNEL(fused_moe_gather_grad, ops::FusedMOEGatherOpGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
                        ops::FusedMOEGatherOpGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
                        ops::FusedMOEGatherOpGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
                        ops::FusedMOEGatherOpGradCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>,
                        ops::FusedMOEGatherOpGradCUDAKernel<paddle::platform::CUDADeviceContext, plat::float16>);

