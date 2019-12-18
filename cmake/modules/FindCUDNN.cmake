include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/cudnn CACHE PATH "Folder contains NVIDIA cuDNN")

set(CUDNN_DIR ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} /usr /usr/local)

find_path(CUDNN_INCLUDE_DIRS
          NAMES cudnn.h
          PATHS ${CUDNN_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          NO_DEFAULT_PATH)

find_library(CUDNN_LIBRARIES
             NAMES cudnn
             PATHS ${CUDNN_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
             NO_DEFAULT_PATH)

set(__looked_for CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)

find_package_handle_standard_args(CUDNN DEFAULT_MSG ${__looked_for})

if (CUDNN_FOUND)
  parse_header(${CUDNN_INCLUDE_DIRS}/cudnn.h
               CUDNN_MAJOR CUDNN_MINOR CUDNN_PATCHLEVEL)
  if (NOT CUDNN_MAJOR)
    set(CUDNN_VERSION "?")
  else ()
    set(CUDNN_VERSION "${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}")
  endif ()
  if (NOT CUDNN_FIND_QUIETLY)
    message(STATUS "Found CUDNN: ${CUDNN_INCLUDE_DIRS}, ${CUDNN_LIBRARIES} (found version ${CUDNN_VERSION})")
  endif ()
  mark_as_advanced(CUDNN_ROOT_DIR ${__looked_for})
else ()
  if (CUDNN_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find CUDNN")
  endif ()
endif ()
