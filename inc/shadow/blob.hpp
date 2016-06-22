#ifndef SHADOW_BLOB_HPP
#define SHADOW_BLOB_HPP

#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

#include "shadow/proto/shadow.pb.h"

template <class Dtype>
class BaseBlob {
 public:
  BaseBlob() {}
  explicit BaseBlob(const std::string &name) : name_(name) {}
  explicit BaseBlob(int count, float *data = nullptr) {
    allocate_data(count);
    if (data != nullptr) set_data(data);
  }

  inline const Dtype *data() const { return data_; }
  inline Dtype *mutable_data() { return data_; }

  inline void set_data(const float *data) {
    if (data == nullptr) Fatal("Set data for blob is nullptr!");
#if !defined(USE_CUDA) & !defined(USE_CL)
    memcpy(data_, data, sizeof(float) * count());
    on_gpu_ = false;

#else
    Kernel::WriteBuffer(count(), data, data_);
    on_gpu_ = true;
#endif
  }

  inline void allocate_data(int count) {
#if !defined(USE_CUDA) & !defined(USE_CL)
    data_ = new float[count];
    on_gpu_ = false;

#else
    data_ = Kernel::MakeBuffer(count, nullptr);
    on_gpu_ = true;
#endif
    if (shape_.size() == 0) add_shape(count);
  }

  inline void copy_data(float *out_data) const {
    if (on_gpu_) {
      Kernel::ReadBuffer(count(), data_, out_data);
    } else {
      memcpy(out_data, data_, sizeof(float) * count());
    }
  }

  inline const std::string name() const { return name_; }
  inline void set_name(const std::string &name) { name_ = name; }

  inline const std::vector<int> shape() const { return shape_; }
  inline std::vector<int> *mutable_shape() { return &shape_; }

  inline const int shape(int index) const {
    if (index < 0 || index >= shape_.size())
      Fatal("Index out of blob shape range!");
    return shape_[index];
  }
  inline void set_shape(int index, int value) {
    if (index < 0 || index >= shape_.size())
      Fatal("Index out of blob shape range!");
    shape_[index] = value;
  }
  inline void set_shape(const shadow::BlobShape &shape) {
    shape_.clear();
    for (int i = 0; i < shape.dim_size(); ++i) {
      shape_.push_back(shape.dim(i));
    }
  }
  inline void add_shape(int value) { shape_.push_back(value); }

  inline const int num_axes() const { return shape_.size(); }
  inline const int num() const { return count() / shape(0); }
  inline const int count() const { return count(0); }
  inline const int count(int start_axis) const {
    return count(start_axis, shape_.size());
  }
  inline const int count(int start_axis, int end_axis) const {
    if (start_axis < 0 || start_axis >= end_axis)
      Fatal("Index out of blob shape range!");
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) count *= shape(i);
    return count;
  }

  inline void clear() {
    if (data_ != nullptr) {
#if !defined(USE_CUDA) & !defined(USE_CL)
      delete[] data_;

#else
      Kernel::ReleaseBuffer(data_);
      data_ = nullptr;
#endif
    }
    shape_.clear();
  }

 private:
  Dtype *data_;

  std::string name_;
  std::vector<int> shape_;
  bool on_gpu_;
};

// TODO(jun): fix Blob template structure
typedef BaseBlob<BType> Blob;
typedef std::vector<Blob *> VecBlob;

inline static Blob *find_blob_by_name(const VecBlob &blobs,
                                      const std::string &name) {
  for (int i = 0; i < blobs.size(); ++i) {
    if (!name.compare(blobs.at(i)->name())) return blobs.at(i);
  }
  return nullptr;
}

#endif  // SHADOW_BLOB_HPP
