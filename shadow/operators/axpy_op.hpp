#ifndef SHADOW_OPERATORS_AXPY_OP_HPP
#define SHADOW_OPERATORS_AXPY_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

/**
 * @param Formulation:
 *            F = a * X + Y
 *        Shape info:
 *            a:  N x C          --> bottoms[0]
 *            X:  N x C x H x W  --> bottoms[1]
 *            Y:  N x C x H x W  --> bottoms[2]
 *            F:  N x C x H x W  --> tops[0]
 */
class AxpyOp : public Operator {
 public:
  explicit AxpyOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    CHECK_EQ(bottoms_size(), 3)
        << "This op must have three bottoms: scale, x and y";
  }

  void Reshape() override;
  void Forward() override;
};

namespace Vision {

template <typename T>
void Axpy(const T *scale_data, const T *x_data, const T *y_data,
          const VecInt &in_shape, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_AXPY_OP_HPP
