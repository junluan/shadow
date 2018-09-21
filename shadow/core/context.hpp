#ifndef SHADOW_CORE_CONTEXT_HPP
#define SHADOW_CORE_CONTEXT_HPP

namespace Shadow {

class Context {
 public:
  Context() = default;
  explicit Context(int device_id);
  ~Context();

  void Reset(int device_id);

  void* blas_handle();
  void* cudnn_handle();
  void* nnpack_handle();

 private:
  void *cublas_handle_ = nullptr, *cudnn_handle_ = nullptr;
  void *nnpack_handle_ = nullptr;

  void Init(int device_id);
  void Clear();
};

}  // namespace Shadow

#endif  // SHADOW_CORE_CONTEXT_HPP
