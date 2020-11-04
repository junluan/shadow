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

mark_as_advanced(CUDNN_ROOT_DIR ${__looked_for})
unset(CUDNN_DIR)

if (CUDNN_INCLUDE_DIRS)
  set(__version_files "cudnn.h" "cudnn_version.h")
  foreach (__version_file ${__version_files})
    if (NOT CUDNN_MAJOR AND EXISTS ${CUDNN_INCLUDE_DIRS}/${__version_file})
      parse_header(${CUDNN_INCLUDE_DIRS}/${__version_file}
                   CUDNN_MAJOR CUDNN_MINOR CUDNN_PATCHLEVEL)
    endif ()
  endforeach ()
  if (CUDNN_MAJOR)
    set(CUDNN_VERSION "${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}")
  else ()
    set(CUDNN_VERSION "?")
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
                                  REQUIRED_VARS ${__looked_for}
                                  VERSION_VAR CUDNN_VERSION)
