#include "blob.hpp"

#include "kernel.hpp"

namespace Shadow {

template <typename T>
void Blob<T>::set_data(const T *data, int set_count, int offset) {
  CHECK_NOTNULL(data);
  CHECK_LE(set_count + offset, count());
#if defined(USE_CUDA)
  Kernel::WriteBuffer(set_count, data, data_ + offset);

#else
  if (!shared_) {
    memcpy(data_ + offset, data, set_count * sizeof(T));
  } else {
    CHECK_EQ(offset, 0);
    data_ = const_cast<T *>(data);
  }
#endif
}

template <typename T>
void Blob<T>::get_data(T *data, int get_count, int offset) const {
  CHECK_NOTNULL(data);
  CHECK_LE(get_count + offset, count());
#if defined(USE_CUDA)
  Kernel::ReadBuffer(get_count, data_ + offset, data);

#else
  memcpy(data, data_ + offset, get_count * sizeof(T));
#endif
}

template <typename T>
void Blob<T>::clear() {
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
  capacity_ = 0;
  shared_ = false;
}

template <typename T>
void Blob<T>::allocate_data(size_t count, bool shared, int align) {
  capacity_ = count;
#if defined(USE_CUDA)
  data_ = Kernel::MakeBuffer<T>(count, static_cast<T *>(nullptr));

#else
  if (!shared) {
    data_ = static_cast<T *>(fast_malloc(count * sizeof(T), align));
  }
  shared_ = shared;
#endif
}

template class Blob<int>;
template class Blob<float>;
template class Blob<unsigned char>;

}  // namespace Shadow
