#ifndef SHADOW_CORE_CONTEXT_HPP
#define SHADOW_CORE_CONTEXT_HPP

namespace Shadow {

class Context {
 public:
  Context() = default;
  explicit Context(int device_id);
  ~Context();

  void Reset(int device_id);
  void SwitchDevice();

  void* blas_handle();
  void* cudnn_handle();
  void* nnpack_handle();

 private:
  int device_id_ = 0;
  void* blas_handle_ = nullptr;
  void* cudnn_handle_ = nullptr;
  void* nnpack_handle_ = nullptr;

  void Init(int device_id);
  void Clear();
};

}  // namespace Shadow

#endif  // SHADOW_CORE_CONTEXT_HPP
