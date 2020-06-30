#include "core/operator.hpp"

namespace Shadow {

class ReshapeOp : public Operator {
 public:
  ReshapeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);
    num_axes_ = get_single_argument<int>("num_axes", -1);
    CHECK_GE(num_axes_, -1);
    shape_ = get_repeated_argument<int>("shape");
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int inferred_axis = -1, constant_count = 1;
    VecInt copy_axes;
    for (int d = 0; d < shape_.size(); ++d) {
      int out_dim = shape_[d];
      if (out_dim == 0) {
        copy_axes.push_back(d);
      } else if (out_dim == -1) {
        CHECK_EQ(inferred_axis, -1);
        inferred_axis = d;
      } else {
        constant_count *= out_dim;
      }
    }

    int start_axis = input->canonical_index(axis_);
    CHECK_GE(start_axis, 0);
    CHECK_LT(start_axis, input->num_axes());
    int end_axis =
        (num_axes_ == -1) ? input->num_axes() : (start_axis + num_axes_);
    CHECK_LE(end_axis, input->num_axes());
    int num_axes_replaced = end_axis - start_axis;
    int num_axes_retained = input->num_axes() - num_axes_replaced;

    VecInt out_shape(num_axes_retained + shape_.size());
    int out_shape_index = 0;
    for (int d = 0; d < start_axis; ++d) {
      out_shape[out_shape_index++] = input->shape(d);
    }
    for (auto dim : shape_) {
      out_shape[out_shape_index++] = dim;
    }
    for (int d = end_axis; d < input->num_axes(); ++d) {
      out_shape[out_shape_index++] = input->shape(d);
    }
    CHECK_EQ(out_shape_index, out_shape.size());

    for (auto d : copy_axes) {
      CHECK_GT(input->num_axes(), start_axis + d);
      out_shape[start_axis + d] = input->shape(start_axis + d);
    }

    if (inferred_axis >= 0) {
      int explicit_count = constant_count;
      explicit_count *= input->count(0, start_axis);
      explicit_count *= input->count(end_axis);
      for (auto d : copy_axes) {
        explicit_count *= out_shape[start_axis + d];
      }
      CHECK_EQ(0, input->count() % explicit_count);
      out_shape[start_axis + inferred_axis] = input->count() / explicit_count;
    }

    output->share_data(input->data<void>(), out_shape);
    CHECK_EQ(output->count(), input->count());
  }

 private:
  int axis_, num_axes_;
  VecInt shape_;
};

REGISTER_OPERATOR(Reshape, ReshapeOp);

}  // namespace Shadow
