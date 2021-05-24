set(main_arch)
set(known_archs 30 32 35 37 50 52 53 60 61 62 70 72 75 80 86)
set(known_archs9 30 32 35 37 50 52 53 60 61 62 70 72)
set(known_archs10 30 32 35 37 50 52 53 60 61 62 70 72 75)
set(known_archs11 52 53 60 61 62 70 72 75 80)
set(known_archs11x 52 53 60 61 62 70 72 75 80 86)

function (detect_cuda_archs cuda_archs)
  if (NOT CUDA_gpu_detect_output)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${__cufile} ""
      "#include <cuda_runtime.h>\n"
      "#include <cstdio>\n"
      "int main() {\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device) {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "    }\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "-ccbin=${CUDA_HOST_COMPILER}" ${CUDA_NVCC_FLAGS} "--run" "${__cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (__nvcc_res EQUAL 0)
      set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architectures from detect_cuda_archs tool" FORCE)
    endif ()
  endif ()

  if (NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    set(${cuda_archs} ${main_arch} PARENT_SCOPE)
  else ()
    set(${cuda_archs} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif ()
endfunction ()

function (select_nvcc_arch_flags out_variable)
  if (${CUDA_VERSION_MAJOR} EQUAL 11)
    if (${CUDA_VERSION_MINOR} GREATER 0)
      set(__known_archs ${known_archs11x})
    else ()
      set(__known_archs ${known_archs11})
    endif ()
  elseif (${CUDA_VERSION_MAJOR} EQUAL 10)
    set(__known_archs ${known_archs10})
  elseif (${CUDA_VERSION_MAJOR} EQUAL 9)
    set(__known_archs ${known_archs9})
  else ()
    set(__known_archs ${known_archs})
  endif ()

  list(GET __known_archs 0 main_arch)

  detect_cuda_archs(__cuda_arch)

  string(REGEX REPLACE "\\." "" __cuda_arch "${__cuda_arch}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch "${__cuda_arch}")
  list_unique(__cuda_arch)

  set(__nvcc_ptx_archs "")
  set(__nvcc_archs_readable "")
  foreach (__arch ${__cuda_arch})
    list(FIND __known_archs ${__arch} __arch_index)
    if (${__arch_index} GREATER -1)
      set(__nvcc_ptx_archs "${__nvcc_ptx_archs},sm_${__arch}")
      list(APPEND __nvcc_archs_readable sm_${__arch})
    endif ()
  endforeach ()

  set(__nvcc_flags "-gencode arch=compute_${main_arch},code=\\\"compute_${main_arch}${__nvcc_ptx_archs}\\\"")

  set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction ()
