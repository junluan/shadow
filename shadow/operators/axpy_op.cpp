#include "axpy_op.hpp"

namespace Shadow {

void AxpyOp::Forward() {
  CHECK_EQ(bottoms_size(), 3)
      << "This op must have three bottoms: scale, x and y";

  const auto *scale = bottoms<float>(0);
  const auto *x = bottoms<float>(1);
  const auto *y = bottoms<float>(2);
  auto *top = mutable_tops<float>(0);

  CHECK_EQ(scale->shape(0), x->shape(0));
  CHECK_EQ(scale->shape(1), x->shape(1));
  if (scale->num_axes() == 4) {
    CHECK_EQ(scale->shape(2), 1);
    CHECK_EQ(scale->shape(3), 1);
  }
  CHECK(x->shape() == y->shape());

  top->reshape(x->shape());

  Vision::Axpy(scale->data(), x->data(), y->data(), x->shape(),
               top->mutable_data());

  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(Axpy, AxpyOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Axpy(const T *scale_data, const T *x_data, const T *y_data,
          const VecInt &in_shape, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = batch * in_c * in_h * in_w;
  int spatial_dim = in_h * in_w;
  Blas::BlasScopy(count, y_data, 0, out_data, 0);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      int scale_offset = b * in_c + c;
      int data_offset = scale_offset * spatial_dim;
      Blas::BlasSaxpy(spatial_dim, scale_data[scale_offset], x_data,
                      data_offset, out_data, data_offset);
    }
  }
}

template void Axpy(const float *scale_data, const float *x_data,
                   const float *y_data, const VecInt &in_shape,
                   float *out_data);

#elif defined(USE_CL)
#endif

}  // namespace Vision

}  // namespace Shadow
