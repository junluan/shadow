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
  Blob(std::string name, Allocator *allocator)
      : name_(std::move(name)), allocator_(allocator) {
    CHECK_NOTNULL(allocator_);
  }
  ~Blob() {
    if (data_ != nullptr && !shared_) {
      allocator_->free(data_);
    }
    data_ = nullptr;
    allocator_ = nullptr;
  }

  const T *data() const { return data_; }
  T *mutable_data() { return data_; }

  const T *cpu_data() {
    if (allocator_->device_type() == DeviceType::kGPU) {
      auto cou = count();
      cpu_data_.resize(cou, 0);
      get_data(cpu_data_.data(), cou);
      return cpu_data_.data();
    } else {
      return data_;
    }
  }

  void set_data(const T *cpu_data, int set_count, int offset = 0) {
    CHECK_NOTNULL(cpu_data);
    CHECK_LE(set_count + offset, count());
    CHECK_NOTNULL(data_);
    allocator_->write(set_count * sizeof(T), cpu_data, data_ + offset);
  }

  void get_data(T *cpu_data, int get_count, int offset = 0) const {
    CHECK_NOTNULL(cpu_data);
    CHECK_LE(get_count + offset, count());
    CHECK_NOTNULL(data_);
    allocator_->read(get_count * sizeof(T), data_ + offset, cpu_data);
  }

  void share_data(const T *data, const std::vector<int> &shape) {
    CHECK_NOTNULL(data);
    if (data_ != nullptr && !shared_) {
      allocator_->free(data_);
    }
    data_ = const_cast<T *>(data);
    shape_ = shape;
    capacity_ = 0;
    shared_ = true;
    CHECK_GT(count(), 0);
  }

  void reshape(const std::vector<int> &shape) {
    size_t cou = 1;
    for (const auto dim : shape) cou *= dim;
    CHECK_GT(cou, 0);
    if (data_ != nullptr && !shared_ && cou > capacity_) {
      allocator_->free(data_);
    }
    if (data_ == nullptr || shared_ || cou > capacity_) {
      data_ = static_cast<T *>(allocator_->malloc(cou * sizeof(T), nullptr));
      capacity_ = cou;
    }
    shape_ = shape;
    shared_ = false;
  }

  const std::string &name() const { return name_; }
  size_t capacity() const { return capacity_; }
  bool shared() const { return shared_; }

  void set_name(const std::string &name) { name_ = name; }
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

 private:
  Allocator *allocator_{nullptr};

  std::string name_{""};
  T *data_{nullptr};
  std::vector<int> shape_{};
  size_t capacity_{0};
  bool shared_{false};

  std::vector<T> cpu_data_{};

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
