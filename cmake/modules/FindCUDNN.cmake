include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")

set(CUDNN_DIR ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

find_path(CUDNN_INCLUDE_DIRS
          NAMES cudnn.h
          PATHS ${CUDNN_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "CUDNN include header cudnn.h" )

find_library(CUDNN_LIBRARIES
             NAMES cudnn
             PATHS ${CUDNN_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "CUDNN library")

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)

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
  mark_as_advanced(CUDNN_ROOT_DIR CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
else ()
  if (CUDNN_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find CUDNN")
  endif ()
endif ()
