#include "connected_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void ConnectedOp::Setup() {
  CHECK(has_argument("num_output"));
  num_output_ = get_single_argument<int>("num_output", 0);
  bias_term_ = get_single_argument<bool>("bias_term", true);
  transpose_ = get_single_argument<bool>("transpose", false);

  if (bias_term_) {
    CHECK_EQ(blobs_size(), 2);
  } else {
    CHECK_EQ(blobs_size(), 1);
  }
}

void ConnectedOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0);

  VecInt top_shape{batch, num_output_};
  top->reshape(top_shape);

  if (bias_term_) {
    biases_multiplier_.reshape(batch);
    Blas::Set(batch, 1, biases_multiplier_.mutable_data(), 0);
  }

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void ConnectedOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0), bottom_num = bottom->num();
  if (batch == 1) {
    Blas::BlasSgemv(0, num_output_, bottom_num, 1, blobs<float>(0)->data(), 0,
                    bottom->data(), 0, 0, top->mutable_data(), 0);
    if (bias_term_) {
      Blas::BlasSaxpy(num_output_, 1, blobs<float>(1)->data(), 0,
                      top->mutable_data(), 0);
    }
  } else {
    Blas::BlasSgemm(0, !transpose_, batch, num_output_, bottom_num, 1,
                    bottom->data(), 0, blobs<float>(0)->data(), 0, 0,
                    top->mutable_data(), 0);
    if (bias_term_) {
      Blas::BlasSgemm(0, 0, batch, num_output_, 1, 1, biases_multiplier_.data(),
                      0, blobs<float>(1)->data(), 0, 1, top->mutable_data(), 0);
    }
  }
}

void ConnectedOp::Release() {
  biases_multiplier_.clear();

  // DLOG(INFO) << "Free ConnectedOp!";
}

REGISTER_OPERATOR(Connected, ConnectedOp);

}  // namespace Shadow
