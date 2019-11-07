#include "gather_op.hpp"

namespace Shadow {

void GatherOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  BlobI *indexes = nullptr;
  if (indexes_value_.empty()) {
    CHECK_EQ(bottoms_size(), 2);
    indexes = const_cast<BlobI *>(bottoms<int>(1));
    CHECK_EQ(indexes->num_axes(), 1);
  } else {
    int num_indexes = static_cast<int>(indexes_value_.size());
    op_ws_->GrowTempBuffer(num_indexes, sizeof(int));
    indexes =
        op_ws_->CreateTempBlob<int>({num_indexes}, op_name_ + "/indexes_value");
    indexes->set_data(indexes_value_.data(), indexes->count());
  }

  int num_indexes = indexes->count();

  CHECK_GT(num_indexes, 0);

  auto top_shape = bottom->shape();
  top_shape[axis_] = num_indexes;
  top->reshape(top_shape);

  int gather_dim = bottom->shape(axis_), inner_num = bottom->count(axis_ + 1);

  Vision::Gather(bottom->data(), indexes->data(), num_indexes, gather_dim,
                 inner_num, top->count(), top->mutable_data());
}

REGISTER_OPERATOR(Gather, GatherOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Gather(const T *in_data, const int *indexes_data, int num_indexes,
            int gather_dim, int inner_num, int count, T *out_data) {
  int gather_num = num_indexes * inner_num;
  for (int i = 0; i < count; ++i) {
    int gather_index = indexes_data[(i / inner_num) % num_indexes];
    int in_index = (gather_index + i / gather_num * gather_dim) * inner_num +
                   i % inner_num;
    out_data[i] = in_data[in_index];
  }
}

template void Gather(const float *, const int *, int, int, int, int, float *);
#endif

}  // namespace Vision

}  // namespace Shadow
