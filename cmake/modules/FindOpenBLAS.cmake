set(OpenBLAS_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/openblas CACHE PATH "Folder contains OpenBLAS")

set(OpenBLAS_DIR ${OpenBLAS_ROOT_DIR} /usr /usr/local /usr/local/opt/openblas)

find_path(OpenBLAS_INCLUDE_DIRS
          NAMES cblas.h
          PATHS ${OpenBLAS_DIR}
          PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu include/openblas
          NO_DEFAULT_PATH)

find_library(OpenBLAS_LIBRARIES
             NAMES openblas libopenblas
             PATHS ${OpenBLAS_DIR}
             PATH_SUFFIXES lib lib/x86_64-linux-gnu lib/aarch64-linux-gnu
             NO_DEFAULT_PATH)

set(__looked_for OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)

mark_as_advanced(OpenBLAS_ROOT_DIR ${__looked_for})
unset(OpenBLAS_DIR)

if (OpenBLAS_INCLUDE_DIRS)
  set(__version_files "openblas_config.h" "openblas/openblas_config.h")
  foreach (__version_file ${__version_files})
    if (NOT OPENBLAS_VERSION AND EXISTS ${OpenBLAS_INCLUDE_DIRS}/${__version_file})
      parse_header_single_define(${OpenBLAS_INCLUDE_DIRS}/${__version_file}
                                 OPENBLAS_VERSION
                                 "[0-9]+\\.[0-9]+\\.[0-9]+")
    endif ()
  endforeach ()
  if (OPENBLAS_VERSION)
    set(OpenBLAS_VERSION ${OPENBLAS_VERSION})
  else ()
    set(OpenBLAS_VERSION "?")
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS
                                  REQUIRED_VARS ${__looked_for}
                                  VERSION_VAR OpenBLAS_VERSION)
