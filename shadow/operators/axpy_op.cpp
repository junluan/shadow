#include "axpy_op.hpp"

namespace Shadow {

void AxpyOp::Forward() {
  CHECK_EQ(bottoms_size(), 3)
      << "This op must have three bottoms: scale, x and y";

  const auto scale = bottoms(0);
  const auto x = bottoms(1);
  const auto y = bottoms(2);
  auto top = tops(0);

  CHECK_EQ(scale->shape(0), x->shape(0));
  CHECK_EQ(scale->shape(1), x->shape(1));
  if (scale->num_axes() == 4) {
    CHECK_EQ(scale->shape(2), 1);
    CHECK_EQ(scale->shape(3), 1);
  }
  CHECK(x->shape() == y->shape());

  top->reshape(x->shape());

  Vision::Axpy(scale->data<float>(), x->data<float>(), y->data<float>(),
               x->shape(), top->mutable_data<float>(), ws_->Ctx());
}

REGISTER_OPERATOR(Axpy, AxpyOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Axpy(const T *scale_data, const T *x_data, const T *y_data,
          const VecInt &in_shape, T *out_data, Context *context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = batch * in_c * in_h * in_w;
  int spatial_dim = in_h * in_w;
  Blas::BlasScopy(count, y_data, 0, out_data, 0, context);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      int scale_offset = b * in_c + c;
      int data_offset = scale_offset * spatial_dim;
      Blas::BlasSaxpy(spatial_dim, scale_data[scale_offset], x_data,
                      data_offset, out_data, data_offset, context);
    }
  }
}

template void Axpy(const float *, const float *, const float *, const VecInt &,
                   float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
