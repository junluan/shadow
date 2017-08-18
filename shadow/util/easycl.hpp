#ifndef SHADOW_UTIL_EASYCL_HPP
#define SHADOW_UTIL_EASYCL_HPP

#include <algorithm>  // std::copy
#include <iostream>   // std::cout
#include <map>        // std::map
#include <memory>     // std::shared_ptr
#include <numeric>    // std::accumulate
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <vector>     // std::vector

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace EasyCL {

enum BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned };

inline void Error(const std::string &message) {
  throw std::runtime_error("Internal OpenCL error: " + message);
}

inline void CheckError(const cl_int status) {
  if (status != CL_SUCCESS) {
    Error(std::to_string(status));
  }
}

class Platform {
 public:
  explicit Platform(const cl_platform_id platform) : platform_(platform) {}
  explicit Platform(int platform_id) {
    auto num_platforms = cl_uint{0};
    CheckError(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0) {
      Error("No platforms found!");
    }
    auto platforms = std::vector<cl_platform_id>(num_platforms);
    CheckError(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
    if (platform_id >= num_platforms) {
      Error("Invalid platform ID " + std::to_string(platform_id));
    }
    platform_ = platforms[platform_id];
  }

  int NumDevices() const {
    auto num_devices = cl_uint{0};
    CheckError(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr,
                              &num_devices));
    return static_cast<int>(num_devices);
  }

  const cl_platform_id &operator()() const { return platform_; }

 private:
  cl_platform_id platform_;
};

class Device {
 public:
  explicit Device(const cl_device_id device) : device_(device) {}
  explicit Device(const Platform &platform, int device_id) {
    auto num_devices = platform.NumDevices();
    if (num_devices == 0) {
      Error("No devices found!");
    }
    auto devices = std::vector<cl_device_id>(num_devices);
    CheckError(clGetDeviceIDs(platform(), CL_DEVICE_TYPE_ALL,
                              static_cast<cl_uint>(num_devices), devices.data(),
                              nullptr));
    if (device_id >= num_devices) {
      Error("Invalid device ID " + std::to_string(device_id));
    }
    device_ = devices[device_id];
  }

  const std::string Version() const { return GetInfoString(CL_DEVICE_VERSION); }
  size_t VersionNumber() const {
    const auto &version_string = Version().substr(7);
    auto next_whitespace = version_string.find(' ');
    auto version = static_cast<size_t>(
        100.0 * std::stod(version_string.substr(0, next_whitespace)));
    return version;
  }
  const std::string Vendor() const { return GetInfoString(CL_DEVICE_VENDOR); }
  const std::string Name() const { return GetInfoString(CL_DEVICE_NAME); }
  const std::string Type() const {
    auto type = GetInfo<cl_device_type>(CL_DEVICE_TYPE);
    switch (type) {
      case CL_DEVICE_TYPE_CPU:
        return "CPU";
      case CL_DEVICE_TYPE_GPU:
        return "GPU";
      case CL_DEVICE_TYPE_ACCELERATOR:
        return "Accelerator";
      default:
        return "default";
    }
  }
  size_t MaxWorkGroupSize() const {
    return GetInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
  }
  size_t MaxWorkItemDimensions() const {
    return static_cast<size_t>(
        GetInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
  }
  const std::vector<size_t> MaxWorkItemSizes() const {
    return GetInfoVector<size_t>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
  }
  unsigned long LocalMemSize() const {
    return static_cast<unsigned long>(
        GetInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE));
  }
  const std::string Capabilities() const {
    return GetInfoString(CL_DEVICE_EXTENSIONS);
  }
  size_t CoreClock() const {
    return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY));
  }
  size_t ComputeUnits() const {
    return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS));
  }
  unsigned long MemorySize() const {
    return static_cast<unsigned long>(
        GetInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE));
  }
  unsigned long MaxAllocSize() const {
    return static_cast<unsigned long>(
        GetInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE));
  }
  size_t MemoryClock() const { return 0; }     // Not exposed in OpenCL
  size_t MemoryBusWidth() const { return 0; }  // Not exposed in OpenCL

  bool IsLocalMemoryValid(const cl_ulong local_mem_usage) const {
    return local_mem_usage <= LocalMemSize();
  }
  bool IsThreadConfigValid(const std::vector<size_t> &local) const {
    auto local_size = size_t{1};
    for (const auto &item : local) {
      local_size *= item;
    }
    for (auto i = size_t{0}; i < local.size(); ++i) {
      if (local[i] > MaxWorkItemSizes()[i]) {
        return false;
      }
    }
    if (local_size > MaxWorkGroupSize()) {
      return false;
    }
    return local.size() <= MaxWorkItemDimensions();
  }

  bool IsCPU() const { return Type() == "CPU"; }
  bool IsGPU() const { return Type() == "GPU"; }
  bool IsNVIDIA() const {
    return Vendor() == "NVIDIA" || Vendor() == "NVIDIA Corporation";
  }
  bool IsAMD() const {
    return Vendor() == "AMD" || Vendor() == "Advanced Micro Devices, Inc." ||
           Vendor() == "AuthenticAMD";
  }
  bool IsIntel() const {
    return Vendor() == "INTEL" || Vendor() == "Intel" ||
           Vendor() == "GenuineIntel";
  }
  bool IsARM() const { return Vendor() == "ARM"; }

  const cl_device_id &operator()() const { return device_; }

 private:
  template <typename T>
  const T GetInfo(const cl_device_info info) const {
    auto bytes = size_t{0};
    CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
    auto result = T(0);
    CheckError(clGetDeviceInfo(device_, info, bytes, &result, nullptr));
    return result;
  }
  template <typename T>
  const std::vector<T> GetInfoVector(const cl_device_info info) const {
    auto bytes = size_t{0};
    CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
    auto result = std::vector<T>(bytes / sizeof(T));
    CheckError(clGetDeviceInfo(device_, info, bytes, result.data(), nullptr));
    return result;
  }
  const std::string GetInfoString(const cl_device_info info) const {
    auto bytes = size_t{0};
    CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
    auto result = std::string();
    result.resize(bytes);
    CheckError(clGetDeviceInfo(device_, info, bytes, &result[0], nullptr));
    return result;
  }

  cl_device_id device_;
};

class Event {
 public:
  explicit Event(const cl_event event) : event_(new cl_event) {
    *event_ = event;
  }
  explicit Event()
      : event_(new cl_event, [](cl_event *e) {
          if (*e) {
            CheckError(clReleaseEvent(*e));
          }
          delete e;
        }) {
    *event_ = nullptr;
  }

  void WaitForCompletion() const { CheckError(clWaitForEvents(1, &(*event_))); }

  float GetElapsedTime() const {
    WaitForCompletion();
    const auto bytes = sizeof(cl_ulong);
    auto time_start = cl_ulong{0};
    clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_START, bytes,
                            &time_start, nullptr);
    auto time_end = cl_ulong{0};
    clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_END, bytes, &time_end,
                            nullptr);
    return static_cast<float>(time_end - time_start) * 1.0e-6f;
  }

  const cl_event &operator()() const { return *event_; }
  const cl_event *pointer() const { return &(*event_); }

  cl_event &operator()() { return *event_; }
  cl_event *pointer() { return &(*event_); }

 private:
  std::shared_ptr<cl_event> event_;
};

class Context {
 public:
  explicit Context(const cl_context context) : context_(new cl_context) {
    *context_ = context;
  }
  explicit Context(const Device &device)
      : context_(new cl_context, [](cl_context *c) {
          CheckError(clReleaseContext(*c));
          delete c;
        }) {
    auto status = CL_SUCCESS;
    const auto dev = device();
    *context_ = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
    CheckError(status);
  }

  const cl_context &operator()() const { return *context_; }
  const cl_context *pointer() const { return &(*context_); }

  cl_context &operator()() { return *context_; }
  cl_context *pointer() { return &(*context_); }

 private:
  std::shared_ptr<cl_context> context_;
};

class Queue {
 public:
  explicit Queue(const cl_command_queue queue) : queue_(new cl_command_queue) {
    *queue_ = queue;
  }
  explicit Queue(const Context &context, const Device &device)
      : queue_(new cl_command_queue, [](cl_command_queue *s) {
          CheckError(clReleaseCommandQueue(*s));
          delete s;
        }) {
    auto status = CL_SUCCESS;
#if defined(CL_VERSION_2_0)
    size_t ocl_version = device.VersionNumber();
    if (ocl_version >= 200) {
      cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES,
                                          CL_QUEUE_PROFILING_ENABLE, 0};
      *queue_ = clCreateCommandQueueWithProperties(context(), device(),
                                                   properties, &status);
    } else {
      *queue_ = clCreateCommandQueue(context(), device(),
                                     CL_QUEUE_PROFILING_ENABLE, &status);
    }
#else
    *queue_ = clCreateCommandQueue(context(), device(),
                                   CL_QUEUE_PROFILING_ENABLE, &status);
#endif
    CheckError(status);
  }

  void Finish(Event & /*unused*/) const { Finish(); }
  void Finish() const { CheckError(clFinish(*queue_)); }

  Context GetContext() const {
    auto bytes = size_t{0};
    CheckError(
        clGetCommandQueueInfo(*queue_, CL_QUEUE_CONTEXT, 0, nullptr, &bytes));
    cl_context result;
    CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_CONTEXT, bytes, &result,
                                     nullptr));
    return Context(result);
  }
  Device GetDevice() const {
    auto bytes = size_t{0};
    CheckError(
        clGetCommandQueueInfo(*queue_, CL_QUEUE_DEVICE, 0, nullptr, &bytes));
    cl_device_id result;
    CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_DEVICE, bytes, &result,
                                     nullptr));
    return Device(result);
  }

  const cl_command_queue &operator()() const { return *queue_; }
  const cl_command_queue *pointer() const { return &(*queue_); }

  cl_command_queue &operator()() { return *queue_; }
  cl_command_queue *pointer() { return &(*queue_); }

 private:
  std::shared_ptr<cl_command_queue> queue_;
};

class Program {
 public:
  explicit Program(const Context &context, const std::string &source)
      : program_(new cl_program,
                 [](cl_program *p) {
                   CheckError(clReleaseProgram(*p));
                   delete p;
                 }),
        length_(source.length()),
        source_(source),
        source_ptr_(&source_[0]) {
    auto status = CL_SUCCESS;
    *program_ = clCreateProgramWithSource(context(), 1, &source_ptr_, &length_,
                                          &status);
    CheckError(status);
  }

  explicit Program(const Device &device, const Context &context,
                   const std::string &binary)
      : program_(new cl_program,
                 [](cl_program *p) {
                   CheckError(clReleaseProgram(*p));
                   delete p;
                 }),
        length_(binary.length()),
        source_(binary),
        source_ptr_(&source_[0]) {
    auto status1 = CL_SUCCESS;
    auto status2 = CL_SUCCESS;
    const cl_device_id dev = device();
    *program_ = clCreateProgramWithBinary(
        context(), 1, &dev, &length_,
        reinterpret_cast<const unsigned char **>(&source_ptr_), &status1,
        &status2);
    CheckError(status1);
    CheckError(status2);
  }

  void Build(const Device &device, const std::vector<std::string> &options) {
    const auto &options_string =
        std::accumulate(options.begin(), options.end(), std::string(" "));
    const auto dev = device();
    auto status = clBuildProgram(*program_, 1, &dev, options_string.c_str(),
                                 nullptr, nullptr);
    if (status != CL_SUCCESS) {
      std::cout << GetBuildInfo(device) << std::endl;
      CheckError(status);
    }
  }

  const std::string GetBuildInfo(const Device &device) const {
    auto bytes = size_t{0};
    auto query = cl_program_build_info{CL_PROGRAM_BUILD_LOG};
    CheckError(
        clGetProgramBuildInfo(*program_, device(), query, 0, nullptr, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(clGetProgramBuildInfo(*program_, device(), query, bytes,
                                     &result[0], nullptr));
    return result;
  }

  const std::string GetIR() const {
    auto bytes = size_t{0};
    CheckError(clGetProgramInfo(*program_, CL_PROGRAM_BINARY_SIZES,
                                sizeof(size_t), &bytes, nullptr));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(clGetProgramInfo(*program_, CL_PROGRAM_BINARIES, sizeof(char *),
                                &result[0], nullptr));
    return result;
  }

  const cl_program &operator()() const { return *program_; }
  const cl_program *pointer() const { return &(*program_); }

 private:
  size_t length_;
  std::string source_;
  const char *source_ptr_;

  std::shared_ptr<cl_program> program_;
};

template <typename T>
class Buffer {
 public:
  explicit Buffer(const cl_mem buffer)
      : buffer_(new cl_mem), access_(BufferAccess::kNotOwned) {
    *buffer_ = buffer;
  }

  explicit Buffer(const Context &context, const BufferAccess access,
                  const size_t size)
      : buffer_(new cl_mem,
                [access](cl_mem *m) {
                  if (access != BufferAccess::kNotOwned) {
                    CheckError(clReleaseMemObject(*m));
                  }
                  delete m;
                }),
        access_(access) {
    auto flags = cl_mem_flags{CL_MEM_READ_WRITE};
    if (access_ == BufferAccess::kReadOnly) {
      flags = CL_MEM_READ_ONLY;
    } else if (access_ == BufferAccess::kWriteOnly) {
      flags = CL_MEM_WRITE_ONLY;
    }
    auto status = CL_SUCCESS;
    *buffer_ =
        clCreateBuffer(context(), flags, size * sizeof(T), nullptr, &status);
    CheckError(status);
  }

  explicit Buffer(const Context &context, const size_t size)
      : Buffer<T>(context, BufferAccess::kReadWrite, size) {}

  template <typename Iterator>
  explicit Buffer(const Context &context, const Queue &queue, Iterator start,
                  Iterator end)
      : Buffer(context, BufferAccess::kReadWrite,
               static_cast<size_t>(end - start)) {
    auto size = static_cast<size_t>(end - start);
    auto pointer = &*start;
    CheckError(clEnqueueWriteBuffer(queue(), *buffer_, CL_FALSE, 0,
                                    size * sizeof(T), pointer, 0, nullptr,
                                    nullptr));
    queue.Finish();
  }

  void ReadAsync(const Queue &queue, const size_t size, T *host,
                 const size_t offset = 0) const {
    if (access_ == BufferAccess::kWriteOnly) {
      Error("Reading from a write-only buffer!");
    }
    CheckError(clEnqueueReadBuffer(queue(), *buffer_, CL_FALSE,
                                   offset * sizeof(T), size * sizeof(T), host,
                                   0, nullptr, nullptr));
  }
  void Read(const Queue &queue, const size_t size, T *host,
            const size_t offset = 0) const {
    ReadAsync(queue, size, host, offset);
    queue.Finish();
  }

  void WriteAsync(const Queue &queue, const size_t size, const T *host,
                  const size_t offset = 0) {
    if (access_ == BufferAccess::kReadOnly) {
      Error("Writing to a read-only buffer!");
    }
    if (GetSize() < (offset + size) * sizeof(T)) {
      Error("Target device buffer is too small!");
    }
    CheckError(clEnqueueWriteBuffer(queue(), *buffer_, CL_FALSE,
                                    offset * sizeof(T), size * sizeof(T), host,
                                    0, nullptr, nullptr));
  }
  void Write(const Queue &queue, const size_t size, const T *host,
             const size_t offset = 0) {
    WriteAsync(queue, size, host, offset);
    queue.Finish();
  }

  void CopyToAsync(const Queue &queue, const size_t size,
                   const Buffer<T> &dest) const {
    CheckError(clEnqueueCopyBuffer(queue(), *buffer_, dest(), 0, 0,
                                   size * sizeof(T), 0, nullptr, nullptr));
  }
  void CopyTo(const Queue &queue, const size_t size,
              const Buffer<T> &dest) const {
    CopyToAsync(queue, size, dest);
    queue.Finish();
  }

  size_t GetSize() const {
    const auto bytes = sizeof(size_t);
    auto result = size_t{0};
    CheckError(
        clGetMemObjectInfo(*buffer_, CL_MEM_SIZE, bytes, &result, nullptr));
    return result;
  }

  const cl_mem &operator()() const { return *buffer_; }
  cl_mem &operator()() { return *buffer_; }

 private:
  const BufferAccess access_;

  std::shared_ptr<cl_mem> buffer_;
};

class Kernel {
 public:
  explicit Kernel(const cl_kernel kernel) : kernel_(new cl_kernel) {
    *kernel_ = kernel;
  }
  explicit Kernel(const Program &program, const std::string &name)
      : kernel_(new cl_kernel, [](cl_kernel *k) {
          CheckError(clReleaseKernel(*k));
          delete k;
        }) {
    auto status = CL_SUCCESS;
    *kernel_ = clCreateKernel(program(), name.c_str(), &status);
    CheckError(status);
  }

  template <typename T>
  void SetArgument(const size_t index, const T &value) {
    CheckError(clSetKernelArg(*kernel_, static_cast<cl_uint>(index), sizeof(T),
                              &value));
  }
  template <typename T>
  void SetArgument(const size_t index, const Buffer<T> &value) {
    SetArgument(index, value());
  }

  template <typename... Args>
  void SetArguments(Args &... args) {
    SetArgumentsRecursive(0, args...);
  }

  unsigned long LocalMemUsage(const Device &device) const {
    const auto bytes = sizeof(cl_ulong);
    auto query = cl_kernel_work_group_info{CL_KERNEL_LOCAL_MEM_SIZE};
    auto result = cl_ulong{0};
    CheckError(clGetKernelWorkGroupInfo(*kernel_, device(), query, bytes,
                                        &result, nullptr));
    return static_cast<unsigned long>(result);
  }

  const std::string GetFunctionName() const {
    auto bytes = size_t{0};
    CheckError(
        clGetKernelInfo(*kernel_, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(clGetKernelInfo(*kernel_, CL_KERNEL_FUNCTION_NAME, bytes,
                               &result[0], nullptr));
    return result;
  }

  void Launch(const Queue &queue, const std::vector<size_t> &global,
              Event *event) {
    CheckError(clEnqueueNDRangeKernel(
        queue(), *kernel_, static_cast<cl_uint>(global.size()), nullptr,
        global.data(), nullptr, 0, nullptr, event->pointer()));
  }

  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, Event *event) {
    CheckError(clEnqueueNDRangeKernel(
        queue(), *kernel_, static_cast<cl_uint>(global.size()), nullptr,
        global.data(), local.data(), 0, nullptr, event->pointer()));
  }

  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, Event *event,
              const std::vector<Event> &wait_for_events) {
    auto wait_for_events_plain = std::vector<cl_event>();
    for (auto &waitEvent : wait_for_events) {
      if (waitEvent()) {
        wait_for_events_plain.push_back(waitEvent());
      }
    }
    CheckError(clEnqueueNDRangeKernel(
        queue(), *kernel_, static_cast<cl_uint>(global.size()), nullptr,
        global.data(), !local.empty() ? local.data() : nullptr,
        static_cast<cl_uint>(wait_for_events_plain.size()),
        !wait_for_events_plain.empty() ? wait_for_events_plain.data() : nullptr,
        event->pointer()));
  }

  const cl_kernel &operator()() const { return *kernel_; }

 private:
  template <typename T>
  void SetArgumentsRecursive(const size_t index, T &first) {
    SetArgument(index, first);
  }
  template <typename T, typename... Args>
  void SetArgumentsRecursive(const size_t index, T &first, Args &... args) {
    SetArgument(index, first);
    SetArgumentsRecursive(index + 1, args...);
  }

  std::shared_ptr<cl_kernel> kernel_;
};

class KernelSet {
 public:
  KernelSet() = default;
  ~KernelSet() {
    for (auto &kernel : cl_kernels_) {
      if (kernel.second != nullptr) {
        delete kernel.second;
        kernel.second = nullptr;
      }
    }
    cl_kernels_.clear();
  }

  void set_kernel(const Program &program, const std::string &kernel_name) {
    if (cl_kernels_.find(kernel_name) != cl_kernels_.end()) {
      throw std::runtime_error("Kernel name " + kernel_name + " is exist!");
    }
    cl_kernels_[kernel_name] = new Kernel(program, kernel_name);
  }

  void set_kernel(const Program &program,
                  const std::vector<std::string> &kernel_names) {
    for (const auto &kernel_name : kernel_names) {
      set_kernel(program, kernel_name);
    }
  }

  Kernel *get_kernel(const std::string &kernel_name) {
    if (cl_kernels_.find(kernel_name) == cl_kernels_.end()) {
      throw std::runtime_error("Kernel " + kernel_name +
                               " is not initialized!");
    }
    return cl_kernels_[kernel_name];
  }

  Kernel *operator[](const std::string &kernel_name) {
    return get_kernel(kernel_name);
  }

 private:
  std::map<std::string, Kernel *> cl_kernels_;
};

}  // namespace EasyCL

namespace EasyCL {

inline std::vector<Platform> GetAllPlatforms() {
  auto num_platforms = cl_uint{0};
  CheckError(clGetPlatformIDs(0, nullptr, &num_platforms));
  auto all_platforms = std::vector<Platform>{};
  for (int platform_id = 0; platform_id < num_platforms; ++platform_id) {
    all_platforms.emplace_back(platform_id);
  }
  return all_platforms;
}

inline EasyCL::Device *CreateForPlatformDeviceIndexes(int platform_id,
                                                      int device_id) {
  const auto &platforms = GetAllPlatforms();
  if (platforms.empty()) {
    Error("No OpenCL platforms available!");
  }
  if (platform_id >= platforms.size()) {
    Error("OpenCL platform index out of range: " + std::to_string(platform_id) +
          " >= " + std::to_string(platforms.size()));
  }
  const auto &platform = platforms[platform_id];
  auto num_devices = platform.NumDevices();
  if (num_devices == 0) {
    Error("No OpenCL devices available for platform index " +
          std::to_string(platform_id));
  }
  if (device_id >= num_devices) {
    Error("OpenCL device index out of range for platform index " +
          std::to_string(platform_id) + " : " + std::to_string(device_id) +
          " >= " + std::to_string(num_devices));
  }
  return new EasyCL::Device(platform, device_id);
}

inline EasyCL::Device *CreateForIndexedGPU(int gpu) {
  const auto &platforms = GetAllPlatforms();
  if (platforms.empty()) {
    Error("No OpenCL platforms available!");
  }
  int current_gpu_index = 0;
  for (const auto &platform : platforms) {
    auto num_devices = cl_uint{0};
    auto status = clGetDeviceIDs(platform(), CL_DEVICE_TYPE_GPU, 0, nullptr,
                                 &num_devices);
    if (status != CL_SUCCESS) continue;
    if ((gpu - current_gpu_index) < num_devices) {
      return new EasyCL::Device(platform, gpu - current_gpu_index);
    }
    current_gpu_index += num_devices;
  }
  if (current_gpu_index == 0) {
    Error("No OpenCL enabled GPUs found!");
  } else {
    Error("Not enough OpenCL enabled GPUs found to satisfy gpu index " +
          std::to_string(gpu));
  }
  return nullptr;
}

inline EasyCL::Device *CreateForFirstGPU() { return CreateForIndexedGPU(0); }

inline EasyCL::Device *CreateForFirstGPUOtherwiseCPU() {
  try {
    return CreateForIndexedGPU(0);
  } catch (std::runtime_error &error) {
    std::cout << error.what() << std::endl;
    std::cout << "Trying for OpenCL enabled CPU" << std::endl;
  }
  return CreateForPlatformDeviceIndexes(0, 0);
}

}  // namespace EasyCL

#endif  // SHADOW_UTIL_EASYCL_HPP
