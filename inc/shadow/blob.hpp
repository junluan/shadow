#ifndef SHADOW_BLOB_HPP
#define SHADOW_BLOB_HPP

#include "shadow/kernel.hpp"
#include "shadow/util/log.hpp"
#include "shadow/util/type.hpp"

#if defined(USE_CL)
#define BACKEND EasyCL::Buffer<Dtype>
#else
#define BACKEND Dtype
#endif

template <typename Dtype>
class Blob {
 public:
  explicit Blob(const std::string &name = "") : name_(name) {}
  ~Blob() { clear(); }

  inline const BACKEND *data() const { return data_; }
  inline BACKEND *mutable_data() { return data_; }

  inline void set_data(const Dtype *data, bool shared = false) {
    CHECK_NOTNULL(data);
#if !defined(USE_CUDA) & !defined(USE_CL)
    if (!shared) {
      memcpy(data_, data, count() * sizeof(Dtype));
    } else {
      data_ = const_cast<Dtype *>(data);
    }
    on_gpu_ = false;
    shared_ = shared;

#else
    Kernel::WriteBuffer(count(), data, data_);
    on_gpu_ = true;
#endif
  }

  inline void read_data(Dtype *data) const {
    CHECK_NOTNULL(data);
#if !defined(USE_CUDA) & !defined(USE_CL)
    memcpy(data, data_, count() * sizeof(Dtype));

#else
    Kernel::ReadBuffer(count(), data_, data);
#endif
  }

  inline void share_data(const BACKEND *data) {
    CHECK_NOTNULL(data);
    if (data_ != data) {
      CHECK(data_ == nullptr);
    }
    data_ = const_cast<BACKEND *>(data);
    shared_ = true;
#if !defined(USE_CUDA) & !defined(USE_CL)
    on_gpu_ = false;

#else
    on_gpu_ = true;
#endif
  }

  inline void reshape(const VecInt &shape, bool shared = false) {
    if (shape.size() == 0) return;
    int new_count = 1;
    for (const auto &dim : shape) new_count *= dim;
    CHECK_GT(new_count, 0);
    if (data_ == nullptr || new_count > count()) {
      clear();
      allocate_data(new_count, shared);
    }
    set_shape(shape);
  }
  inline void reshape(int num, int channels = 1, int height = 1, int width = 1,
                      bool shared = false) {
    VecInt shape(4);
    shape[0] = num;
    shape[1] = channels;
    shape[2] = height;
    shape[3] = width;
    reshape(shape, shared);
  }

  inline const std::string &name() const { return name_; }
  inline std::string &name() { return name_; }

  inline const VecInt &shape() const { return shape_; }
  inline VecInt &shape() { return shape_; }

  inline const int shape(int index) const {
    return shape_[canonical_index(index)];
  }
  inline void set_shape(int index, int value) {
    shape_[canonical_index(index)] = value;
  }
  inline void set_shape(const VecInt &shape) { shape_ = shape; }
  inline void add_shape(int value) { shape_.push_back(value); }

  inline const int num_axes() const { return shape_.size(); }
  inline const int num() const { return count() / shape(0); }
  inline const int count() const { return count(0); }
  inline const int count(int start_axis) const {
    return count(start_axis, num_axes());
  }
  inline const int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) count *= shape(i);
    return count;
  }

  inline const int canonical_index(int index) const {
    CHECK_GE(index, -num_axes());
    CHECK_LT(index, num_axes());
    if (index < 0) {
      return index + num_axes();
    }
    return index;
  }

  inline void clear() {
    if (data_ != nullptr && !shared_) {
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
  inline void allocate_data(int count, bool shared) {
#if !defined(USE_CUDA) & !defined(USE_CL)
    if (!shared) {
      data_ = new Dtype[count];
    }
    on_gpu_ = false;
    shared_ = shared;

#else
    data_ = Kernel::MakeBuffer<BACKEND>(count, static_cast<Dtype *>(nullptr));
    on_gpu_ = true;
#endif
  }

  BACKEND *data_ = nullptr;

  std::string name_ = "";
  VecInt shape_;
  bool on_gpu_ = false;
  bool shared_ = false;
};

typedef std::vector<Blob<float> *> VecBlob;

inline static Blob<float> *get_blob_by_name(const VecBlob &blobs,
                                            const std::string &name) {
  for (const auto &blob : blobs) {
    if (!name.compare(blob->name())) return blob;
  }
  return nullptr;
}

#endif  // SHADOW_BLOB_HPP
