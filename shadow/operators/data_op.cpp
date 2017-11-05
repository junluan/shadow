#include "data_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

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

  Vision::DataTransform(bottom->data(), bottom->shape(), scale_, num_mean_,
                        mean_value_->data(), top->mutable_data());
}

REGISTER_OPERATOR(Data, DataOp);

}  // namespace Shadow
