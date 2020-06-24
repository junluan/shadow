#include "core/operator.hpp"

#include "kernels/grid_sample.hpp"

namespace Shadow {

class GridSampleOp : public Operator {
 public:
  GridSampleOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    // Nearest: 0, Bilinear: 1
    mode_ = get_single_argument<int>("mode", 1);
    CHECK(mode_ == 0 || mode_ == 1);
    // Zeros: 0, Border: 1
    padding_mode_ = get_single_argument<int>("padding_mode", 0);
    CHECK(padding_mode_ == 0 || padding_mode_ == 1);

    kernel_ = std::dynamic_pointer_cast<GridSampleKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Forward() override {
    CHECK_EQ(bottoms_size(), 2);

    const auto bottom = bottoms(0);
    const auto grid = bottoms(1);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    CHECK_EQ(bottom->shape(0), grid->shape(0));
    CHECK_EQ(grid->shape(3), 2);

    auto top_shape = bottom->shape();
    top_shape[2] = grid->shape(1);
    top_shape[3] = grid->shape(2);
    top->reshape(top_shape);

    kernel_->Run(bottom, grid, top, ws_, mode_, padding_mode_);
  }

 private:
  int mode_, padding_mode_;

  std::shared_ptr<GridSampleKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(GridSample, GridSampleOp);

}  // namespace Shadow
