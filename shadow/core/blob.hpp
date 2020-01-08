#ifndef SHADOW_CORE_BLOB_HPP
#define SHADOW_CORE_BLOB_HPP

#include "allocator.hpp"
#include "common.hpp"

#include "util/log.hpp"

#include <string>
#include <vector>

namespace Shadow {

template <typename T>
class Blob {
 public:
  explicit Blob(const std::string &name = "") : name_(name) { set_device(); }
  explicit Blob(const std::vector<int> &shape, const std::string &name = "",
                bool shared = false)
      : Blob(name) {
    reshape(shape, shared);
  }
  ~Blob() { clear(); }

  const T *data() const { return data_; }
  T *mutable_data() { return data_; }

  const T *cpu_data() {
#if defined(USE_CUDA)
    auto cou = count();
    cpu_data_.resize(cou, 0);
    get_data(cpu_data_.data(), cou);
    return cpu_data_.data();

#else
    return data_;
#endif
  }

  void set_data(const T *data, int set_count, int offset = 0) {
    CHECK_NOTNULL(data);
    CHECK_LE(set_count + offset, count());
#if defined(USE_CUDA)
    Allocator::WriteBuffer<T>(set_count, data, data_ + offset);

#else
    if (!shared_) {
      Allocator::WriteBuffer<T>(set_count, data, data_ + offset);
    } else {
      CHECK_EQ(offset, 0);
      data_ = const_cast<T *>(data);
    }
#endif
  }

  void get_data(T *data, int get_count, int offset = 0) const {
    CHECK_NOTNULL(data);
    CHECK_LE(get_count + offset, count());
    Allocator::ReadBuffer<T>(get_count, data_ + offset, data);
  }

  void share_data(const T *data, const std::vector<int> &shape) {
    CHECK_NOTNULL(data);
    size_t cou = 1;
    for (const auto dim : shape) cou *= dim;
    CHECK_EQ(cou, count());
    if (data_ != data) {
      CHECK(data_ == nullptr);
    }
    data_ = const_cast<T *>(data);
    shared_ = true;
  }
  void share_data(const Blob<T> &from) { share_data(from.data_, from.shape_); }

  void reshape(const std::vector<int> &shape, bool shared = false,
               int align = 1) {
    size_t cou = 1;
    for (const auto dim : shape) cou *= dim;
    CHECK_GT(cou, 0);
    if (data_ == nullptr || cou > capacity_) {
      clear();
      allocate_data(cou, shared, align);
    }
    set_shape(shape);
  }

  const std::string &name() const { return name_; }
  Device device() const { return device_; }
  size_t capacity() const { return capacity_; }
  bool shared() const { return shared_; }

  void set_name(const std::string &name) { name_ = name; }
  void set_device(Device device) { device_ = device; }
  void set_capacity(size_t capacity) { capacity_ = capacity; }
  void set_shared(bool shared) { shared_ = shared; }

  const std::vector<int> &shape() const { return shape_; }
  int shape(int index) const { return shape_[canonical_index(index)]; }
  void set_shape(int index, int value) {
    shape_[canonical_index(index)] = value;
  }
  void set_shape(const std::vector<int> &shape) { shape_ = shape; }
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

  size_t mem_count() const { return capacity_ * sizeof(T); }

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
      Allocator::ReleaseBuffer<T>(data_);
    }
    data_ = nullptr;
    cpu_data_.clear();
    shape_.clear();
    capacity_ = 0;
    shared_ = false;
  }

 private:
  void allocate_data(size_t count, bool shared, int align) {
    capacity_ = count;
#if defined(USE_CUDA)
    data_ = static_cast<T *>(Allocator::MakeBuffer<T>(count, nullptr, align));

#else
    if (!shared) {
      data_ = static_cast<T *>(Allocator::MakeBuffer<T>(count, nullptr, align));
    }
    shared_ = shared;
#endif
  }

  void set_device() {
#if defined(USE_CUDA)
    device_ = Device::kGPU;

#else
    device_ = Device::kCPU;
#endif
  }

  T *data_ = nullptr;
  std::vector<T> cpu_data_{};

  std::string name_;
  std::vector<int> shape_{};
  Device device_ = Device::kCPU;
  size_t capacity_ = 0;
  bool shared_ = false;

  DISABLE_COPY_AND_ASSIGN(Blob<T>);
};

using BlobI = Blob<int>;
using BlobF = Blob<float>;
using BlobUC = Blob<unsigned char>;

using VecBlobI = std::vector<BlobI *>;
using VecBlobF = std::vector<BlobF *>;
using VecBlobUC = std::vector<BlobUC *>;

}  // namespace Shadow

#endif  // SHADOW_CORE_BLOB_HPP
