include(FindPackageHandleStandardArgs)

set(OpenBLAS_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/openblas CACHE PATH "Folder contains OpenBLAS")

set(OpenBLAS_DIR ${OpenBLAS_ROOT_DIR} /usr /usr/local /usr/local/opt/openblas)

find_path(OpenBLAS_INCLUDE_DIRS
          NAMES cblas.h
          PATHS ${OpenBLAS_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64 include/x86_64-linux-gnu include/aarch64-linux-gnu
          NO_DEFAULT_PATH)

find_library(OpenBLAS_LIBRARIES
             NAMES openblas libopenblas
             PATHS ${OpenBLAS_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
             NO_DEFAULT_PATH)

set(__looked_for OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)

find_package_handle_standard_args(OpenBLAS DEFAULT_MSG ${__looked_for})

if (OpenBLAS_FOUND)
  set(__version_files "openblas_config.h" "openblas/openblas_config.h")
  foreach (__version_file ${__version_files})
    if (EXISTS ${OpenBLAS_INCLUDE_DIRS}/${__version_file})
      parse_header_single_define(${OpenBLAS_INCLUDE_DIRS}/${__version_file}
                                 OPENBLAS_VERSION
                                 "[0-9]+\\.[0-9]+\\.[0-9]+")
    endif ()
  endforeach ()
  if (NOT OPENBLAS_VERSION)
    set(OpenBLAS_VERSION "?")
  else ()
    set(OpenBLAS_VERSION ${OPENBLAS_VERSION})
  endif ()
  if (NOT OpenBLAS_FIND_QUIETLY)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_INCLUDE_DIRS}, ${OpenBLAS_LIBRARIES} (found version ${OpenBLAS_VERSION})")
  endif ()
  mark_as_advanced(OpenBLAS_ROOT_DIR ${__looked_for})
else ()
  if (OpenBLAS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find OpenBLAS")
  endif ()
endif ()
