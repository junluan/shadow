#ifndef SHADOW_OPERATORS_DATA_OP_HPP
#define SHADOW_OPERATORS_DATA_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DataOp : public Operator {
 public:
  DataOp() {}
  explicit DataOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~DataOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  float scale_;
  int num_mean_;

  BlobF mean_value_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DATA_OP_HPP
