#include "core/operator.hpp"

#include "kernels/slice.hpp"

namespace Shadow {

class SliceOp : public Operator {
 public:
  SliceOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    slice_point_ = get_repeated_argument<int>("slice_point");

    kernel_ = std::dynamic_pointer_cast<SliceKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_GE(outputs.size(), 2);

    const auto& input = inputs[0];

    axis_ = input->canonical_index(axis_);

    int num_outputs = static_cast<int>(outputs.size());

    VecInt slices;
    int in_slice_axis = input->shape(axis_);
    if (slice_point_.empty()) {
      CHECK_EQ(in_slice_axis % num_outputs, 0);
      slices.resize(num_outputs, in_slice_axis / num_outputs);
    } else {
      CHECK_EQ(slice_point_.size(), num_outputs - 1);
      int prev = 0;
      for (auto point : slice_point_) {
        CHECK_GT(point, prev);
        slices.push_back(point - prev);
        prev = point;
      }
      CHECK_GT(in_slice_axis, prev);
      slices.push_back(in_slice_axis - prev);
    }

    int count = 0;
    auto out_shape = input->shape();
    for (int n = 0; n < num_outputs; ++n) {
      auto& output = outputs[n];
      out_shape[axis_] = slices[n];
      output->reshape(out_shape);
      count += output->count();
    }
    CHECK_EQ(count, input->count());

    kernel_->Run(input, outputs, ws_, axis_);
  }

 private:
  int axis_;
  VecInt slice_point_;

  std::shared_ptr<SliceKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Slice, SliceOp);

}  // namespace Shadow
