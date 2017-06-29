#include "data_op.hpp"
#include "core/image.hpp"

namespace Shadow {

void DataOp::Setup() {
  bottoms_.push_back(op_ws_->GetBlob("in_blob"));

  scale_ = arg_helper_.GetSingleArgument<float>("scale", 1);
  VecFloat mean_value = arg_helper_.GetRepeatedArgument<float>("mean_value");
  num_mean_ = 1;
  if (mean_value.size() > 1) {
    CHECK_EQ(mean_value.size(), bottoms_[0]->shape(1));
    num_mean_ = mean_value.size();
  } else if (mean_value.size() == 0) {
    mean_value.push_back(0);
  }
  mean_value_.reshape(num_mean_);
  mean_value_.set_data(mean_value.data(), num_mean_);
}

void DataOp::Reshape() {
  tops_[0]->reshape(bottoms_[0]->shape());

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")");
}

void DataOp::Forward() {
  Image::DataTransform(bottoms_[0]->data(), bottoms_[0]->shape(), scale_,
                       num_mean_, mean_value_.data(), tops_[0]->mutable_data());
}

void DataOp::Release() {
  mean_value_.clear();

  // DLOG(INFO) << "Free DataOp!";
}

REGISTER_OP_CLASS(Data);

}  // namespace Shadow
