#include "depthwise_conv_op.hpp"

#include "activate_op.hpp"

namespace Shadow {

void DepthwiseConvOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int in_h = bottom->shape(2), in_w = bottom->shape(3);

  VecInt top_shape = bottom->shape();
  top_shape[1] = num_output_;
  top_shape[2] =
      depthwise_conv_out_size(in_h, kernel_size_, stride_, pad_, dilation_);
  top_shape[3] =
      depthwise_conv_out_size(in_w, kernel_size_, stride_, pad_, dilation_);
  top->reshape(top_shape);

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << num_output_ << "_" << kernel_size_ << "x" << kernel_size_
             << "_s" << stride_ << "_p" << pad_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void DepthwiseConvOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bias_term_) {
    Vision::DepthwiseConv(bottom->data(), bottom->shape(),
                          blobs<float>(0)->data(), blobs<float>(1)->data(),
                          kernel_size_, stride_, pad_, bias_term_, top->shape(),
                          top->mutable_data());
  } else {
    Vision::DepthwiseConv(bottom->data(), bottom->shape(),
                          blobs<float>(0)->data(), blobs<float>(0)->data(),
                          kernel_size_, stride_, pad_, bias_term_, top->shape(),
                          top->mutable_data());
  }

  if (activate_type_ == 1) {
    Vision::Activate(top->mutable_data(), top->count(), activate_type_);
  }
}

REGISTER_OPERATOR(DepthwiseConv, DepthwiseConvOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void DepthwiseConv(const T *in_data, const VecInt &in_shape,
                   const T *weight_data, const T *bias_data, int kernel_size,
                   int stride, int pad, int bias_term, const VecInt &out_shape,
                   T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      const T *in_offset_data = in_data + (b * in_c + c) * in_h * in_w;
      const T *weight_offset_data = weight_data + c * kernel_size * kernel_size;
      T *out_offset_data = out_data + (b * in_c + c) * out_h * out_w;
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int hstart = h * stride - pad, wstart = w * stride - pad;
          int hend = std::min(hstart + kernel_size, in_h + pad);
          int wend = std::min(wstart + kernel_size, in_w + pad);
          hstart = std::max(hstart, 0), wstart = std::max(wstart, 0);
          hend = std::min(hend, in_h), wend = std::min(wend, in_w);
          int khstart = hend < kernel_size ? (kernel_size - hend) : 0;
          int kwstart = wend < kernel_size ? (kernel_size - wend) : 0;
          auto sum_val = T(0);
          for (int kh = hstart; kh < hend; ++kh) {
            for (int kw = wstart; kw < wend; ++kw) {
              sum_val +=
                  in_offset_data[kh * in_w + kw] *
                  weight_offset_data[(khstart + kh - hstart) * kernel_size +
                                     kwstart + kw - wstart];
            }
          }
          if (bias_term) {
            sum_val += bias_data[c];
          }
          out_offset_data[h * out_w + w] = sum_val;
        }
      }
    }
  }
}

template void DepthwiseConv(const float *in_data, const VecInt &in_shape,
                            const float *weight_data, const float *bias_data,
                            int kernel_size, int stride, int pad, int bias_term,
                            const VecInt &out_shape, float *out_data);

#elif defined(USE_CL)
#endif

}  // namespace Vision

}  // namespace Shadow
