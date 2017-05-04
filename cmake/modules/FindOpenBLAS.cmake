include(FindPackageHandleStandardArgs)

set(OpenBLAS_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/openblas CACHE PATH "Folder contains OpenBLAS")

set(OpenBLAS_DIR ${OpenBLAS_ROOT_DIR} /usr /usr/local)

find_path(OpenBLAS_INCLUDE_DIRS
          NAMES cblas.h
          PATHS ${OpenBLAS_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "OpenBLAS include header cblas.h"
          NO_DEFAULT_PATH)

find_library(OpenBLAS_LIBRARIES
             NAMES openblas libopenblas.dll
             PATHS ${OpenBLAS_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "OpenBLAS library"
             NO_DEFAULT_PATH)

find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)

if (NOT OpenBLAS_INCLUDE_DIRS AND NOT OpenBLAS_FIND_QUIETLY)
  message(STATUS "Could not find OpenBLAS include")
endif ()

if (NOT OpenBLAS_LIBRARIES AND NOT OpenBLAS_FIND_QUIETLY)
  message(STATUS "Could not find OpenBLAS lib")
endif ()

if (OpenBLAS_FOUND)
  if (NOT OpenBLAS_FIND_QUIETLY)
    message(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIRS}")
    message(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIBRARIES}")
  endif ()
  mark_as_advanced(OpenBLAS_ROOT_DIR OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)
else ()
  if (OpenBLAS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find OpenBLAS")
  endif ()
endif ()
