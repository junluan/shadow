#include "connected_op.hpp"

namespace Shadow {

void ConnectedOp::Forward() {
  if (bias_term_) {
    CHECK_EQ(bottoms_size(), 3);
  } else {
    CHECK_EQ(bottoms_size(), 2);
  }

  const auto bottom = bottoms(0);
  const auto weight = bottoms(1);
  auto top = tops(0);

  int batch = bottom->shape(0), bottom_num = bottom->num();

  VecInt top_shape{batch, num_output_};
  top->reshape(top_shape);

  if (batch == 1) {
    if (transpose_) {
      Blas::BlasSgemv(0, num_output_, bottom_num, 1, weight->data<float>(), 0,
                      bottom->data<float>(), 0, 0, top->mutable_data<float>(),
                      0, ws_->Ctx());
    } else {
      Blas::BlasSgemv(1, bottom_num, num_output_, 1, weight->data<float>(), 0,
                      bottom->data<float>(), 0, 0, top->mutable_data<float>(),
                      0, ws_->Ctx());
    }
    if (bias_term_) {
      Blas::BlasSaxpy(num_output_, 1, bottoms(2)->data<float>(), 0,
                      top->mutable_data<float>(), 0, ws_->Ctx());
    }
  } else {
    Blas::BlasSgemm(0, transpose_, batch, num_output_, bottom_num, 1,
                    bottom->data<float>(), 0, weight->data<float>(), 0, 0,
                    top->mutable_data<float>(), 0, ws_->Ctx());
    if (bias_term_) {
      ws_->GrowTempBuffer(batch * sizeof(float));
      auto biases_multiplier = ws_->CreateTempBlob({batch}, DataType::kF32);
      Blas::Set(batch, 1, biases_multiplier->mutable_data<float>(), 0,
                ws_->Ctx());
      Blas::BlasSgemm(0, 0, batch, num_output_, 1, 1,
                      biases_multiplier->data<float>(), 0,
                      bottoms(2)->data<float>(), 0, 1,
                      top->mutable_data<float>(), 0, ws_->Ctx());
    }
  }
}

REGISTER_OPERATOR(Connected, ConnectedOp);

}  // namespace Shadow
