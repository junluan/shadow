include(FindPackageHandleStandardArgs)

set(NNPACK_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/nnpack CACHE PATH "Folder contains NNPACK")

set(NNPACK_DIR ${NNPACK_ROOT_DIR} /usr /usr/local)

find_path(NNPACK_INCLUDE_DIRS
          NAMES nnpack.h
          PATHS ${NNPACK_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "NNPACK include header nnpack.h"
          NO_DEFAULT_PATH)

find_library(NNPACK_LIBRARY
             NAMES nnpack
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "NNPACK library"
             NO_DEFAULT_PATH)

find_library(PTHREADPOOL_LIBRARY
             NAMES pthreadpool
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "NNPACK library"
             NO_DEFAULT_PATH)

find_package_handle_standard_args(NNPACK DEFAULT_MSG NNPACK_INCLUDE_DIRS NNPACK_LIBRARY PTHREADPOOL_LIBRARY)

if (NNPACK_FOUND)
  set(NNPACK_LIBRARIES "${NNPACK_LIBRARY};${PTHREADPOOL_LIBRARY}")
  if (NOT NNPACK_FIND_QUIETLY)
    message(STATUS "Found NNPACK: ${NNPACK_INCLUDE_DIRS}, ${NNPACK_LIBRARIES}")
  endif ()
  mark_as_advanced(NNPACK_ROOT_DIR NNPACK_INCLUDE_DIRS NNPACK_LIBRARY PTHREADPOOL_LIBRARY)
else ()
  if (NNPACK_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find NNPACK")
  endif ()
endif ()
