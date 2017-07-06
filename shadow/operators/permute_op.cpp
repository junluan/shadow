#include "permute_op.hpp"
#include "core/image.hpp"

namespace Shadow {

void PermuteOp::Setup() {
  permute_order_data_ = arg_helper_.GetRepeatedArgument<int>("order");
  num_axes_ = permute_order_data_.size();
  CHECK_EQ(num_axes_, bottoms_[0]->num_axes());
}

void PermuteOp::Reshape() {
  VecInt top_shape, old_steps(num_axes_), new_steps(num_axes_);
  for (const auto &order : permute_order_data_) {
    top_shape.push_back(bottoms_[0]->shape(order));
  }
  tops_[0]->reshape(top_shape);

  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps[i] = 1;
      new_steps[i] = 1;
    } else {
      old_steps[i] = bottoms_[0]->count(i + 1);
      new_steps[i] = tops_[0]->count(i + 1);
    }
  }

  permute_order_.reshape(num_axes_);
  old_steps_.reshape(num_axes_);
  new_steps_.reshape(num_axes_);

  permute_order_.set_data(permute_order_data_.data(), num_axes_);
  old_steps_.set_data(old_steps.data(), num_axes_);
  new_steps_.set_data(new_steps.data(), num_axes_);

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void PermuteOp::Forward() {
  Image::Permute(bottoms_[0]->data(), bottoms_[0]->count(),
                 bottoms_[0]->num_axes(), permute_order_.data(),
                 old_steps_.data(), new_steps_.data(),
                 tops_[0]->mutable_data());
}

void PermuteOp::Release() {
  permute_order_.clear();
  old_steps_.clear();
  new_steps_.clear();

  // DLOG(INFO) << "Free PermuteOp!";
}

REGISTER_OPERATOR(Permute, PermuteOp);

}  // namespace Shadow
