#include "permute_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void PermuteOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  VecInt top_shape, old_steps(num_axes_), new_steps(num_axes_);
  for (const auto &order : permute_order_data_) {
    top_shape.push_back(bottom->shape(order));
  }
  top->reshape(top_shape);

  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps[i] = 1;
      new_steps[i] = 1;
    } else {
      old_steps[i] = bottom->count(i + 1);
      new_steps[i] = top->count(i + 1);
    }
  }

  permute_order_ = op_ws_->CreateBlob<int>(op_name_ + "_permute_order");
  old_steps_ = op_ws_->CreateBlob<int>(op_name_ + "_old_steps");
  new_steps_ = op_ws_->CreateBlob<int>(op_name_ + "_new_steps");

  permute_order_->reshape({num_axes_});
  old_steps_->reshape({num_axes_});
  new_steps_->reshape({num_axes_});

  permute_order_->set_data(permute_order_data_.data(), num_axes_);
  old_steps_->set_data(old_steps.data(), num_axes_);
  new_steps_->set_data(new_steps.data(), num_axes_);

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << Util::format_vector(permute_order_data_, ",", "(", ")")
             << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void PermuteOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Permute(bottom->data(), bottom->count(), bottom->num_axes(),
                  permute_order_->data(), old_steps_->data(),
                  new_steps_->data(), top->mutable_data());
}

REGISTER_OPERATOR(Permute, PermuteOp);

}  // namespace Shadow
