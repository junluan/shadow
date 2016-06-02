#ifndef SHADOW_BLOB_HPP
#define SHADOW_BLOB_HPP

#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

#include "shadow/proto/shadow.pb.h"

#include <vector>

template <class Dtype> class Blob {
public:
  inline const Dtype *data() { return (const Dtype *)data_; }
  inline Dtype *mutable_data() { return data_; }

  inline void set_data(float *data) {
#if defined(USE_CUDA)
    CUDA::CUDAWriteBuffer(count_, data_, data);
    on_gpu_ = true;
#elif defined(USE_CL)
    CL::CLWriteBuffer(count_, data_, data);
    on_gpu_ = true;
#else
    memcpy(data_, data, sizeof(float) * count_);
    on_gpu_ = false;
#endif
  }

  inline void allocate_data(int count) {
#if defined(USE_CUDA)
    data_ = CUDA::CUDAMakeBuffer(count, NULL);
    on_gpu_ = true;
#elif defined(USE_CL)
    data_ = new cl_mem();
    *data_ = CL::CLMakeBuffer(count, CL_MEM_READ_WRITE, nullptr);
    on_gpu_ = true;
#else
    data_ = new float[count];
    on_gpu_ = false;
#endif
    count_ = count;
  }

  inline void copy_data(float *out_data) const {
    if (on_gpu_) {
#if defined(USE_CUDA)
      CUDA::CUDAReadBuffer(count_, data_, out_data);
#elif defined(USE_CL)
      CL::CLReadBuffer(count_, data_, out_data);
#endif
    } else {
      memcpy(out_data, data_, sizeof(float) * count_);
    }
  }

  inline const std::vector<int> shape() { return shape_; }
  inline std::vector<int> *mutable_shape() { return &shape_; }

  inline const int shape(int index) {
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

  inline void clear() {
    if (data_ != nullptr) {
#if defined(USE_CUDA)
      CUDA::CUDAReleaseBuffer(data_);
#elif defined(USE_CL)
      CL::CLReleaseBuffer(data_);
#else
      delete[] data_;
#endif
    }
    shape_.clear();
    num_ = -1;
    count_ = -1;
  }

private:
  Dtype *data_;

  std::vector<int> shape_;
  int num_, count_;
  bool on_gpu_;
};

#endif // SHADOW_BLOB_HPP
