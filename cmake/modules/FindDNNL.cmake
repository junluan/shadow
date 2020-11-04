set(DNNL_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/dnnl CACHE PATH "Folder contains Intel DNNL")

set(DNNL_DIR ${DNNL_ROOT_DIR}/build /usr /usr/local)

find_path(DNNL_INCLUDE_DIRS
          NAMES dnnl.hpp
          PATHS ${DNNL_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          NO_DEFAULT_PATH)

find_library(DNNL_LIBRARIES
             NAMES dnnl
             PATHS ${DNNL_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64
             NO_DEFAULT_PATH)

set(__looked_for DNNL_INCLUDE_DIRS DNNL_LIBRARIES)

mark_as_advanced(DNNL_ROOT_DIR ${__looked_for})
unset(DNNL_DIR)

if (DNNL_INCLUDE_DIRS)
  parse_header(${DNNL_INCLUDE_DIRS}/dnnl_version.h
               DNNL_VERSION_MAJOR DNNL_VERSION_MINOR DNNL_VERSION_PATCH)
  if (DNNL_VERSION_MAJOR)
    set(DNNL_VERSION "${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}.${DNNL_VERSION_PATCH}")
  else ()
    set(DNNL_VERSION "?")
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DNNL
                                  REQUIRED_VARS ${__looked_for}
                                  VERSION_VAR DNNL_VERSION)
