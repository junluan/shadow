#include "data_op.hpp"
#include "core/image.hpp"

namespace Shadow {

void DataOp::Setup() {
  add_bottoms<float>("in_blob");

  scale_ = get_single_argument<float>("scale", 1);
  VecFloat mean_value = get_repeated_argument<float>("mean_value");
  num_mean_ = 1;
  if (mean_value.size() > 1) {
    CHECK_EQ(mean_value.size(), bottoms<float>(0)->shape(1));
    num_mean_ = static_cast<int>(mean_value.size());
  } else if (mean_value.empty()) {
    mean_value.push_back(0);
  }
  mean_value_.reshape(num_mean_);
  mean_value_.set_data(mean_value.data(), num_mean_);
}

void DataOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  top->reshape(bottom->shape());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void DataOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Image::DataTransform(bottom->data(), bottom->shape(), scale_, num_mean_,
                       mean_value_.data(), top->mutable_data());
}

void DataOp::Release() {
  mean_value_.clear();

  // DLOG(INFO) << "Free DataOp!";
}

REGISTER_OPERATOR(Data, DataOp);

}  // namespace Shadow
