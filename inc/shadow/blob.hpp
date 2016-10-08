#ifndef SHADOW_BLOB_HPP
#define SHADOW_BLOB_HPP

#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

#include "shadow/proto/shadow.pb.h"

#if defined(USE_CL)
#define BACKEND cl_mem
#else
#define BACKEND Dtype
#endif

template <typename Dtype>
class Blob {
 public:
  Blob() {}
  explicit Blob(const std::string &name) : name_(name) {}
  explicit Blob(int count, Dtype *data = nullptr) {
    allocate_data(count);
    if (data != nullptr) set_data(data);
  }

  inline const BACKEND *data() const { return data_; }
  inline BACKEND *mutable_data() { return data_; }

  inline void set_data(const Dtype *data) {
    if (data == nullptr) Fatal("Set data for blob is nullptr!");
#if !defined(USE_CUDA) & !defined(USE_CL)
    memcpy(data_, data, count() * sizeof(Dtype));
    on_gpu_ = false;

#else
    Kernel::WriteBuffer(count(), data, data_);
    on_gpu_ = true;
#endif
  }

  inline void allocate_data(int count) {
#if !defined(USE_CUDA) & !defined(USE_CL)
    data_ = new Dtype[count];
    on_gpu_ = false;

#else
    data_ = Kernel::MakeBuffer<BACKEND>(count, static_cast<Dtype *>(nullptr));
    on_gpu_ = true;
#endif
    if (shape_.size() == 0) add_shape(count);
  }

  inline void copy_data(Dtype *out_data) const {
#if !defined(USE_CUDA) & !defined(USE_CL)
    memcpy(out_data, data_, count() * sizeof(Dtype));

#else
    Kernel::ReadBuffer(count(), data_, out_data);
#endif
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
    return count(start_axis, shape_.size() - 1);
  }
  inline const int count(int start_axis, int end_axis) const {
    if (start_axis < 0 || end_axis >= shape_.size() || start_axis > end_axis)
      Fatal("Index out of blob shape range!");
    int count = 1;
    for (int i = start_axis; i <= end_axis; ++i) count *= shape(i);
    return count;
  }

  inline void clear() {
    if (data_ != nullptr) {
#if !defined(USE_CUDA) & !defined(USE_CL)
      delete[] data_;

#else
      Kernel::ReleaseBuffer(data_);
#endif
    }
    data_ = nullptr;
    shape_.clear();
  }

 private:
  BACKEND *data_;

  std::string name_;
  std::vector<int> shape_;
  bool on_gpu_;
};

typedef std::vector<Blob<float> *> VecBlob;

inline static Blob<float> *find_blob_by_name(const VecBlob &blobs,
                                             const std::string &name) {
  for (int i = 0; i < blobs.size(); ++i) {
    if (!name.compare(blobs.at(i)->name())) return blobs.at(i);
  }
  return nullptr;
}

#endif  // SHADOW_BLOB_HPP
