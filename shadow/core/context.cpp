#include "context.hpp"

#include "common.hpp"

#include "util/log.hpp"

namespace Shadow {

class CPUContext : public Context {
 public:
  explicit CPUContext(const ArgumentHelper& arguments) {
    device_id_ = arguments.GetSingleArgument<int>("device_id", 0);

    check_device(device_id_);

#if defined(USE_NNPACK)
    CHECK_EQ(nnp_initialize(), nnp_status_success);
    nnpack_handle_ = pthreadpool_create(0);
    CHECK_NOTNULL(nnpack_handle_);
#endif

#if defined(USE_DNNL)
    dnnl_engine_ = std::make_shared<dnnl::engine>(
        dnnl::engine::kind::cpu, static_cast<size_t>(device_id_));
    dnnl_stream_ = std::make_shared<dnnl::stream>(*dnnl_engine_);
#endif

    allocator_ = GetAllocator<DeviceType::kCPU>();
  }
  ~CPUContext() override {
#if defined(USE_NNPACK)
    if (nnpack_handle_ != nullptr) {
      CHECK_EQ(nnp_deinitialize(), nnp_status_success);
      pthreadpool_destroy(pthreadpool_t(nnpack_handle_));
      nnpack_handle_ = nullptr;
    }
#endif
  }

  Allocator* allocator() const override { return allocator_.get(); }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  int device_id() const override { return device_id_; }

  void switch_device() override {}

  void synchronize() override {}

#if defined(USE_NNPACK)
  void* nnpack_handle() const override {
    CHECK_NOTNULL(nnpack_handle_);
    return nnpack_handle_;
  }
#endif

#if defined(USE_DNNL)
  void* dnnl_engine() const override {
    CHECK_NOTNULL(dnnl_engine_);
    return dnnl_engine_.get();
  }

  void* dnnl_stream() const override {
    CHECK_NOTNULL(dnnl_stream_);
    return dnnl_stream_.get();
  }
#endif

 private:
  static void check_device(int device_id) {
#if defined(USE_DNNL)
    auto num_devices = dnnl::engine::get_count(dnnl::engine::kind::cpu);
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, num_devices);
#endif
  }

  int device_id_ = 0;

  std::shared_ptr<Allocator> allocator_ = nullptr;

#if defined(USE_NNPACK)
  pthreadpool_t nnpack_handle_ = nullptr;
#endif

#if defined(USE_DNNL)
  std::shared_ptr<dnnl::engine> dnnl_engine_ = nullptr;
  std::shared_ptr<dnnl::stream> dnnl_stream_ = nullptr;
#endif
};

template <>
std::shared_ptr<Context> GetContext<DeviceType::kCPU>(
    const ArgumentHelper& arguments) {
  return std::make_shared<CPUContext>(arguments);
}

}  // namespace Shadow
