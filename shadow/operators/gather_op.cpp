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

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    axis_ = bottom->canonical_index(axis_);

    std::shared_ptr<Blob> indexes = nullptr;
    if (indexes_value_.empty()) {
      CHECK_EQ(bottoms_size(), 2);
      indexes = bottoms(1);
      CHECK_EQ(indexes->num_axes(), 1);
    } else {
      int num_indexes = static_cast<int>(indexes_value_.size());
      ws_->GrowTempBuffer(num_indexes * sizeof(int));
      indexes = ws_->CreateTempBlob({num_indexes}, DataType::kI32);
      indexes->set_data<int>(indexes_value_.data(), indexes->count());
    }

    int num_indexes = indexes->count();

    CHECK_GT(num_indexes, 0);

    auto top_shape = bottom->shape();
    top_shape[axis_] = num_indexes;
    top->reshape(top_shape);

    kernel_->Run(bottom, indexes, top, ws_, axis_);
  }

 private:
  int axis_;
  VecInt indexes_value_;

  std::shared_ptr<GatherKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Gather, GatherOp);

}  // namespace Shadow
