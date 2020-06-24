#ifndef SHADOW_CORE_BLOB_HPP_
#define SHADOW_CORE_BLOB_HPP_

#include "allocator.hpp"
#include "common.hpp"

#include "util/log.hpp"

#include <numeric>
#include <string>
#include <vector>

namespace Shadow {

enum class DataType { kI32, kF32, kU8 };

class Blob {
 public:
  Blob(std::string name, DataType data_type, Allocator* allocator)
      : name_(std::move(name)), data_type_(data_type), allocator_(allocator) {
    CHECK_NOTNULL(allocator_);
  }
  ~Blob() {
    if (data_ != nullptr && !shared_) {
      allocator_->free(data_);
    }
    data_ = nullptr;
    allocator_ = nullptr;
  }

  template <typename T>
  const T* data() const {
    return const_cast<const T*>(static_cast<T*>(data_));
  }

  template <typename T>
  T* mutable_data() {
    return const_cast<T*>(data<T>());
  }

  template <typename T>
  const T* cpu_data() {
    check_data_type<T>(data_type_);
    if (allocator_->device_type() == DeviceType::kGPU) {
      cpu_data_.resize(raw_size(), 0);
      get_data<T>(cpu_data_.data(), count());
      return const_cast<const T*>(
          static_cast<T*>(static_cast<void*>(cpu_data_.data())));
    } else {
      return data<T>();
    }
  }

  template <typename T>
  void set_data(const void* cpu_data, int set_count, int offset = 0) {
    check_data_type<T>(data_type_);
    CHECK_NOTNULL(cpu_data);
    CHECK_LE(set_count + offset, count());
    CHECK_NOTNULL(data_);
    allocator_->write(set_count * elem_size(), cpu_data,
                      mutable_data<T>() + offset);
  }

  template <typename T>
  void get_data(void* cpu_data, int get_count, int offset = 0) const {
    check_data_type<T>(data_type_);
    CHECK_NOTNULL(cpu_data);
    CHECK_LE(get_count + offset, count());
    CHECK_NOTNULL(data_);
    allocator_->read(get_count * elem_size(), data<T>() + offset, cpu_data);
  }

  void share_data(const void* data, const std::vector<int>& shape) {
    CHECK_NOTNULL(data);
    if (data_ != nullptr && !shared_) {
      allocator_->free(data_);
    }
    data_ = const_cast<void*>(data);
    shape_ = shape;
    capacity_ = 0;
    shared_ = true;
    CHECK_GT(count(), 0);
  }

  void reshape(const std::vector<int>& shape) {
    auto cou = std::accumulate(shape.begin(), shape.end(), 1,
                               std::multiplies<size_t>());
    CHECK_GT(cou, 0);
    if (data_ != nullptr && !shared_ && cou > capacity_) {
      allocator_->free(data_);
      data_ = nullptr;
    }
    if (data_ == nullptr || shared_) {
      data_ = allocator_->malloc(cou * elem_size(), nullptr);
      capacity_ = cou;
    }
    shape_ = shape;
    shared_ = false;
  }

  const std::string& name() const { return name_; }
  const DataType& data_type() const { return data_type_; }
  size_t capacity() const { return capacity_; }
  bool shared() const { return shared_; }

  void set_name(const std::string& name) { name_ = name; }
  void set_data_type(const DataType& data_type) { data_type_ = data_type; }
  void set_capacity(size_t capacity) { capacity_ = capacity; }
  void set_shared(bool shared) { shared_ = shared; }

  const std::vector<int>& shape() const { return shape_; }
  int shape(int index) const { return shape_[canonical_index(index)]; }
  void set_shape(int index, int value) {
    shape_[canonical_index(index)] = value;
  }
  void set_shape(const std::vector<int>& shape) { shape_ = shape; }
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

  size_t raw_size() const { return count() * elem_size(); }
  size_t max_size() const { return capacity_ * elem_size(); }
  size_t elem_size() const {
    if (data_type_ == DataType::kI32) {
      return sizeof(int);
    } else if (data_type_ == DataType::kF32) {
      return sizeof(float);
    } else if (data_type_ == DataType::kU8) {
      return sizeof(unsigned char);
    } else {
      return 0;
    }
  }

  int canonical_index(int index) const {
    CHECK_GE(index, -num_axes());
    CHECK_LT(index, num_axes());
    if (index < 0) {
      return index + num_axes();
    }
    return index;
  }

 private:
  template <typename T>
  static void check_data_type(DataType data_type) {
    if (std::is_same<T, int>::value) {
      CHECK(data_type == DataType::kI32);
    } else if (std::is_same<T, float>::value) {
      CHECK(data_type == DataType::kF32);
    } else if (std::is_same<T, unsigned char>::value) {
      CHECK(data_type == DataType::kU8);
    } else {
      LOG(FATAL) << "Invalid template typename " << typeid(T).name();
    }
  }

  std::string name_;
  DataType data_type_;
  Allocator* allocator_;

  void* data_{nullptr};
  std::vector<int> shape_{};
  size_t capacity_{0};
  bool shared_{false};

  std::vector<unsigned char> cpu_data_{};

  DISABLE_COPY_AND_ASSIGN(Blob);
};

}  // namespace Shadow

#endif  // SHADOW_CORE_BLOB_HPP_
