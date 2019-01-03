include(FindPackageHandleStandardArgs)

set(NNPACK_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/nnpack CACHE PATH "Folder contains NNPACK")

set(NNPACK_DIR ${NNPACK_ROOT_DIR}/build /usr /usr/local)

find_path(NNPACK_INCLUDE_DIRS
          NAMES nnpack.h
          PATHS ${NNPACK_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          NO_DEFAULT_PATH)

find_library(NNPACK_LIBRARY
             NAMES nnpack
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             NO_DEFAULT_PATH)

find_library(PTHREADPOOL_LIBRARY
             NAMES pthreadpool
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             NO_DEFAULT_PATH)

find_library(CPUINFO_LIBRARY
             NAMES cpuinfo
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             NO_DEFAULT_PATH)

find_library(CLOG_LIBRARY
             NAMES clog
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             NO_DEFAULT_PATH)

set(__looked_for NNPACK_INCLUDE_DIRS NNPACK_LIBRARY PTHREADPOOL_LIBRARY CPUINFO_LIBRARY CLOG_LIBRARY)
set(NNPACK_LIBRARIES ${NNPACK_LIBRARY} ${PTHREADPOOL_LIBRARY} ${CPUINFO_LIBRARY} ${CLOG_LIBRARY})

find_package_handle_standard_args(NNPACK DEFAULT_MSG ${__looked_for})

if (NNPACK_FOUND)
  if (NOT NNPACK_FIND_QUIETLY)
    message(STATUS "Found NNPACK: ${NNPACK_INCLUDE_DIRS}, ${NNPACK_LIBRARIES}")
  endif ()
  mark_as_advanced(NNPACK_ROOT_DIR NNPACK_INCLUDE_DIRS NNPACK_LIBRARY PTHREADPOOL_LIBRARY CPUINFO_LIBRARY CLOG_LIBRARY)
else ()
  if (NNPACK_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find NNPACK")
  endif ()
endif ()
