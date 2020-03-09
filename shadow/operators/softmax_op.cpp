#include "softmax_op.hpp"

namespace Shadow {

void SoftmaxOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

  int axis = bottom->canonical_index(axis_);

  int outer_num = bottom->count(0, axis), channels = bottom->shape(axis),
      inner_num = bottom->count(axis + 1);

  top->reshape(bottom->shape());

#if defined(USE_CUDNN)
  cudnn::setTensor4dDesc<float>(&bottom_top_desc_, outer_num, channels,
                                inner_num, 1);

  CUDNN_CHECK(cudnnSoftmaxForward(
      cudnnHandle_t(ws_->Ctx()->cudnn_handle()), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL, cudnn::dataType<float>::one, bottom_top_desc_,
      bottom->data<float>(), cudnn::dataType<float>::zero, bottom_top_desc_,
      top->mutable_data<float>()));

#else
  ws_->GrowTempBuffer(outer_num * inner_num * sizeof(float));

  auto scalar = ws_->CreateTempBlob({outer_num, inner_num}, DataType::kF32);

  Vision::Softmax(bottom->data<float>(), outer_num, channels, inner_num,
                  scalar->mutable_data<float>(), top->mutable_data<float>(),
                  ws_->Ctx());
#endif
}

REGISTER_OPERATOR(Softmax, SoftmaxOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Softmax(const T *in_data, int outer_num, int channels, int inner_num,
             T *val_data, T *out_data, Context *context) {
  int val_count = outer_num * inner_num, count = val_count * channels;

  for (int i = 0; i < val_count; ++i) {
    int n = i / inner_num, s = i % inner_num;
    const T *in_data_offset = in_data + n * channels * inner_num + s;
    T max_val = T(-FLT_MAX);
    for (int c = 0; c < channels; ++c, in_data_offset += inner_num) {
      max_val = std::max(*in_data_offset, max_val);
    }
    val_data[i] = max_val;
  }

  for (int i = 0; i < count; ++i) {
    int n = i / channels / inner_num, s = i % inner_num;
    out_data[i] = std::exp(in_data[i] - val_data[n * inner_num + s]);
  }

  for (int i = 0; i < val_count; ++i) {
    int n = i / inner_num, s = i % inner_num;
    const T *out_data_offset = out_data + n * channels * inner_num + s;
    T sum = T(0);
    for (int c = 0; c < channels; ++c, out_data_offset += inner_num) {
      sum += *out_data_offset;
    }
    val_data[i] = sum;
  }

  for (int i = 0; i < count; ++i) {
    int n = i / channels / inner_num, s = i % inner_num;
    out_data[i] /= val_data[n * inner_num + s];
  }
}

template void Softmax(const float *, int, int, int, float *, float *,
                      Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
