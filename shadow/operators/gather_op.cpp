#include "gather_op.hpp"

namespace Shadow {

void GatherOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

  CHECK_NE(bottom, top);

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

  int gather_dim = bottom->shape(axis_), inner_num = bottom->count(axis_ + 1);

  Vision::Gather(bottom->data<float>(), indexes->data<int>(), num_indexes,
                 gather_dim, inner_num, top->count(),
                 top->mutable_data<float>(), ws_->Ctx());
}

REGISTER_OPERATOR(Gather, GatherOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Gather(const T *in_data, const int *indexes_data, int num_indexes,
            int gather_dim, int inner_num, int count, T *out_data,
            Context *context) {
  int gather_num = num_indexes * inner_num;
  for (int i = 0; i < count; ++i) {
    int gather_index = indexes_data[(i / inner_num) % num_indexes];
    int in_index = (gather_index + i / gather_num * gather_dim) * inner_num +
                   i % inner_num;
    out_data[i] = in_data[in_index];
  }
}

template void Gather(const float *, const int *, int, int, int, int, float *,
                     Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
