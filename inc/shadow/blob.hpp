#ifndef SHADOW_BLOB_HPP
#define SHADOW_BLOB_HPP

#include "shadow/util/util.hpp"

#include "shadow/proto/shadow.pb.h"

#include <vector>

class Blob {
public:
  inline const float *data() { return (const float *)data_; }
  inline float *mutable_data() { return data_; }

  const std::vector<int> shape() { return shape_; }
  std::vector<int> *mutable_shape() { return &shape_; }

  const inline int shape(int index) {
    if (index < 0 || index >= shape_.size())
      Fatal("Index out of blob shape range!");
    return shape_[index];
  }
  inline void set_shape(int index, int value) {
    if (index < 0 || index >= shape_.size())
      Fatal("Index out of blob shape range!");
    shape_[index] = value;
  }
  inline void add_shape(int value) { shape_.push_back(value); }

  inline const int num() { return num_; }
  inline void set_num(int num) { num_ = num; }
  inline const int count() { return count_; }
  inline void set_count(int count) { count_ = count; }

private:
  float *data_;
  std::vector<int> shape_;
  int num_, count_;
};

#endif // SHADOW_BLOB_HPP
