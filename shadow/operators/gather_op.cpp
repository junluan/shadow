#include "core/operator.hpp"

#include "kernels/gather.hpp"

namespace Shadow {

class GatherOp : public Operator {
 public:
  GatherOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);
    indexes_value_ = get_repeated_argument<int>("indexes");

    kernel_ = std::dynamic_pointer_cast<GatherKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    axis_ = input->canonical_index(axis_);

    std::shared_ptr<Blob> indexes = nullptr;
    if (indexes_value_.empty()) {
      CHECK_EQ(inputs.size(), 2);
      indexes = inputs[1];
      CHECK_EQ(indexes->num_axes(), 1);
    } else {
      int num_indexes = static_cast<int>(indexes_value_.size());
      ws_->GrowTempBuffer(num_indexes * sizeof(int));
      indexes = ws_->CreateTempBlob({num_indexes}, DataType::kI32);
      indexes->set_data<int>(indexes_value_.data(), indexes->count());
    }

    int num_indexes = indexes->count();

    CHECK_GT(num_indexes, 0);

    auto out_shape = input->shape();
    out_shape[axis_] = num_indexes;
    output->reshape(out_shape);

    kernel_->Run(input, indexes, output, ws_, axis_);
  }

 private:
  int axis_;
  VecInt indexes_value_;

  std::shared_ptr<GatherKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Gather, GatherOp);

}  // namespace Shadow
