#include "connected_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void ConnectedOp::Setup(VecBlobF *blobs) {
  Operator::Setup(blobs);

  const auto &conn_param = op_param_.connected_param();

  CHECK(conn_param.has_num_output());
  num_output_ = conn_param.num_output();
  bias_term_ = conn_param.bias_term();
  transpose_ = conn_param.transpose();

  if (bias_term_) {
    CHECK_EQ(blobs_.size(), 2);
  } else {
    CHECK_EQ(blobs_.size(), 1);
  }
}

void ConnectedOp::Reshape() {
  VecInt top_shape;
  top_shape.push_back(bottoms_[0]->shape(0));
  top_shape.push_back(num_output_);
  tops_[0]->reshape(top_shape);

  if (bias_term_) {
    biases_multiplier_.reshape(top_shape[0]);
    Blas::Set(top_shape[0], 1, biases_multiplier_.mutable_data(), 0);
  }

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ConnectedOp::Forward() {
  int batch = bottoms_[0]->shape(0), bottom_num = bottoms_[0]->num();
  if (batch == 1) {
    Blas::BlasSgemv(0, num_output_, bottom_num, 1, blobs_[0]->data(), 0,
                    bottoms_[0]->data(), 0, 0, tops_[0]->mutable_data(), 0);
    if (bias_term_) {
      Blas::BlasSaxpy(num_output_, 1, blobs_[1]->data(), 0,
                      tops_[0]->mutable_data(), 0);
    }
  } else {
    Blas::BlasSgemm(0, !transpose_, batch, num_output_, bottom_num, 1,
                    bottoms_[0]->data(), 0, blobs_[0]->data(), 0, 0,
                    tops_[0]->mutable_data(), 0);
    if (bias_term_) {
      Blas::BlasSgemm(0, 0, batch, num_output_, 1, 1, biases_multiplier_.data(),
                      0, blobs_[1]->data(), 0, 1, tops_[0]->mutable_data(), 0);
    }
  }
}

void ConnectedOp::Release() {
  biases_multiplier_.clear();

  // DLOG(INFO) << "Free ConnectedOp!";
}

}  // namespace Shadow
