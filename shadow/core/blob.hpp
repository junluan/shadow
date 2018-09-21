#ifndef SHADOW_CORE_BLOB_HPP
#define SHADOW_CORE_BLOB_HPP

#include "common.hpp"
#include "kernel.hpp"

#include "util/log.hpp"
#include "util/type.hpp"

namespace Shadow {

enum Device { kCPU, kGPU };

template <typename Dtype>
class Blob {
 public:
  explicit Blob(const std::string &name = "") : name_(name) { set_device(); }
  explicit Blob(const VecInt &shape, const std::string &name = "",
                bool shared = false)
      : name_(name) {
    reshape(shape, shared);
    set_device();
  }
  ~Blob() { clear(); }

  const Dtype *data() const { return data_; }
  Dtype *mutable_data() { return data_; }

  const Dtype *cpu_data() {
#if defined(USE_CUDA)
    auto cou = count();
    cpu_data_.resize(cou, 0);
    read_data(cpu_data_.data(), cou);
    return cpu_data_.data();

#else
    return data_;
#endif
  }

  void set_data(const Dtype *data, int set_count) {
    CHECK_NOTNULL(data);
    CHECK_EQ(set_count, count());
#if defined(USE_CUDA)
    Kernel::WriteBuffer(count(), data, data_);

#else
    if (!shared_) {
      memcpy(data_, data, count() * sizeof(Dtype));
    } else {
      data_ = const_cast<Dtype *>(data);
    }
#endif
  }

  void read_data(Dtype *data, int read_count) const {
    CHECK_NOTNULL(data);
    CHECK_EQ(read_count, count());
#if defined(USE_CUDA)
    Kernel::ReadBuffer(count(), data_, data);

#else
    memcpy(data, data_, count() * sizeof(Dtype));
#endif
  }

  void share_data(const Dtype *data, const VecInt &shape) {
    CHECK_NOTNULL(data);
    size_t cou = 1;
    for (const auto dim : shape) cou *= dim;
    CHECK_EQ(cou, count());
    if (data_ != data) {
      CHECK(data_ == nullptr);
    }
    data_ = const_cast<Dtype *>(data);
    shared_ = true;
  }
  void share_data(const Blob<Dtype> &from) {
    share_data(from.data_, from.shape_);
  }

  void reshape(const VecInt &shape, bool shared = false, int align = 1) {
    size_t cou = 1;
    for (const auto dim : shape) cou *= dim;
    CHECK_GT(cou, 0);
    if (data_ == nullptr || cou > capacity_) {
      clear();
      allocate_data(cou, shared, align);
    }
    set_shape(shape);
  }
  void reshape(int batch, int channel, int height, int width,
               bool shared = false, int align = MALLOC_ALIGN) {
    auto cstep =
        align_size(height * width * sizeof(Dtype), align) / sizeof(Dtype);
    auto cou = batch * channel * cstep;
    CHECK_GT(cou, 0);
    if (data_ == nullptr || cou > capacity_) {
      clear();
      allocate_data(cou, shared, align);
    }
    set_shape(VecInt{batch, channel, height, width});
    set_cstep(cstep);
  }

  const std::string &name() const { return name_; }
  Device device() const { return device_; }
  size_t cstep() const { return cstep_; }
  size_t capacity() const { return capacity_; }
  bool shared() const { return shared_; }

  void set_name(const std::string &name) { name_ = name; }
  void set_device(Device device) { device_ = device; }
  void set_cstep(size_t cstep) { cstep_ = cstep; }
  void set_capacity(size_t capacity) { capacity_ = capacity; }
  void set_shared(bool shared) { shared_ = shared; }

  const VecInt &shape() const { return shape_; }
  int shape(int index) const { return shape_[canonical_index(index)]; }
  void set_shape(int index, int value) {
    shape_[canonical_index(index)] = value;
  }
  void set_shape(const VecInt &shape) { shape_ = shape; }
  void add_shape(int value) { shape_.push_back(value); }

  int num_axes() const { return static_cast<int>(shape_.size()); }
  int num() const { return count() / shape(0); }
  int count() const { return count(0); }
  int count(int start_axis) const { return count(start_axis, num_axes()); }
  int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int cou = 1;
    for (int i = start_axis; i < end_axis; ++i) cou *= shape(i);
    return cou;
  }

  size_t mem_count() const { return capacity_ * sizeof(Dtype); }

  int canonical_index(int index) const {
    CHECK_GE(index, -num_axes());
    CHECK_LT(index, num_axes());
    if (index < 0) {
      return index + num_axes();
    }
    return index;
  }

  void clear() {
    if (data_ != nullptr && !shared_) {
#if defined(USE_CUDA)
      Kernel::ReleaseBuffer(data_);

#else
      fast_free(data_);
#endif
    }
    data_ = nullptr;
    cpu_data_.clear();
    shape_.clear();
    cstep_ = 0;
    capacity_ = 0;
    shared_ = false;
  }

 private:
  void allocate_data(size_t count, bool shared, int align) {
    capacity_ = count;
#if defined(USE_CUDA)
    data_ = Kernel::MakeBuffer<Dtype>(count, static_cast<Dtype *>(nullptr));

#else
    if (!shared) {
      data_ = static_cast<Dtype *>(fast_malloc(count * sizeof(Dtype), align));
    }
    shared_ = shared;
#endif
  }

  void set_device() {
#if defined(USE_CUDA)
    device_ = kGPU;

#else
    device_ = kCPU;
#endif
  }

  Dtype *data_ = nullptr;
  std::vector<Dtype> cpu_data_{};

  std::string name_;
  VecInt shape_{};
  Device device_ = kCPU;
  size_t cstep_ = 0, capacity_ = 0;
  bool shared_ = false;

  DISABLE_COPY_AND_ASSIGN(Blob<Dtype>);
};

using BlobI = Blob<int>;
using BlobF = Blob<float>;
using BlobUC = Blob<unsigned char>;

using VecBlobI = std::vector<BlobI *>;
using VecBlobF = std::vector<BlobF *>;
using VecBlobUC = std::vector<BlobUC *>;

}  // namespace Shadow

#endif  // SHADOW_CORE_BLOB_HPP
