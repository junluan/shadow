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

  void Forward() override {
    CHECK_GE(tops_size(), 2);

    const auto bottom = bottoms(0);

    axis_ = bottom->canonical_index(axis_);

    int num_tops = tops_size();

    VecInt slices;
    int bottom_slice_axis = bottom->shape(axis_);
    if (slice_point_.empty()) {
      CHECK_EQ(bottom_slice_axis % num_tops, 0);
      slices.resize(num_tops, bottom_slice_axis / num_tops);
    } else {
      CHECK_EQ(slice_point_.size(), num_tops - 1);
      int prev = 0;
      for (auto point : slice_point_) {
        CHECK_GT(point, prev);
        slices.push_back(point - prev);
        prev = point;
      }
      CHECK_GT(bottom_slice_axis, prev);
      slices.push_back(bottom_slice_axis - prev);
    }

    std::vector<std::shared_ptr<Blob>> top_blobs;

    int count = 0;
    auto top_shape = bottom->shape();
    for (int n = 0; n < num_tops; ++n) {
      auto top = tops(n);
      top_shape[axis_] = slices[n];
      top->reshape(top_shape);
      count += top->count();
      top_blobs.push_back(top);
    }
    CHECK_EQ(count, bottom->count());

    kernel_->Run(bottom, top_blobs, ws_, axis_);
  }

 private:
  int axis_;
  VecInt slice_point_;

  std::shared_ptr<SliceKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Slice, SliceOp);

}  // namespace Shadow
