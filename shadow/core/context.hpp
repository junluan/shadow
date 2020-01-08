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
  void Synchronize();

  int device_id();

  void* blas_handle();
  void* cudnn_handle();
  void* nnpack_handle();
  void* dnnl_engine();
  void* dnnl_stream();

 private:
  void Init(int device_id);
  void Clear();

  int device_id_ = 0;

  void* blas_handle_ = nullptr;
  void* cudnn_handle_ = nullptr;
  void* nnpack_handle_ = nullptr;
  void* dnnl_engine_ = nullptr;
  void* dnnl_stream_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_CONTEXT_HPP
