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

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), 2);

    const auto& input = inputs[0];
    const auto& grid = inputs[1];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    CHECK_EQ(input->shape(0), grid->shape(0));
    CHECK_EQ(grid->shape(3), 2);

    auto out_shape = input->shape();
    out_shape[2] = grid->shape(1);
    out_shape[3] = grid->shape(2);
    output->reshape(out_shape);

    kernel_->Run(input, grid, output, ws_, mode_, padding_mode_);
  }

 private:
  int mode_, padding_mode_;

  std::shared_ptr<GridSampleKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(GridSample, GridSampleOp);

}  // namespace Shadow
