include(FindPackageHandleStandardArgs)

set(OpenBLAS_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/openblas CACHE PATH "Folder contains OpenBLAS")

set(OpenBLAS_DIR ${OpenBLAS_ROOT_DIR} /usr /usr/local)

set(OpenBLAS_PLATFORM)
set(OpenBLAS_ARC)
if (MSVC)
  set(OpenBLAS_PLATFORM windows)
  set(OpenBLAS_ARC x86_64)
elseif (ANDROID)
  set(OpenBLAS_PLATFORM android)
  set(OpenBLAS_ARC ${ANDROID_ABI})
elseif (APPLE)
  set(OpenBLAS_PLATFORM darwin)
  set(OpenBLAS_ARC x86_64)
elseif (Linux)
  set(OpenBLAS_PLATFORM linux)
  set(OpenBLAS_ARC x86_64)
endif ()

find_path(OpenBLAS_INCLUDE_DIRS
          NAMES cblas.h
          PATHS ${OpenBLAS_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "OpenBLAS include header cblas.h"
          NO_DEFAULT_PATH)

if (MSVC)
  set(TMP ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .dll.a)
endif ()

find_library(OpenBLAS_LIBRARIES
             NAMES openblas libopenblas
             PATHS ${OpenBLAS_DIR}
             PATH_SUFFIXES lib lib/${OpenBLAS_PLATFORM}/${OpenBLAS_ARC} lib64 lib/x86_64 lib/x64 lib/x86
             DOC "OpenBLAS library"
             NO_DEFAULT_PATH)

if (MSVC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${TMP})
endif ()

find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)

if (OpenBLAS_FOUND)
  shadow_parse_header_single_define(${OpenBLAS_INCLUDE_DIRS}/openblas_config.h
                                    OPENBLAS_VERSION
                                    "[0-9]+\\.[0-9]+\\.[0-9]+.[a-z]+")
  if (NOT OPENBLAS_VERSION)
    set(OpenBLAS_VERSION "?")
  else ()
    set(OpenBLAS_VERSION ${OPENBLAS_VERSION})
  endif ()
  if (NOT OpenBLAS_FIND_QUIETLY)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_INCLUDE_DIRS}, ${OpenBLAS_LIBRARIES} (found version ${OpenBLAS_VERSION})")
  endif ()
  mark_as_advanced(OpenBLAS_ROOT_DIR OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)
else ()
  if (OpenBLAS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find OpenBLAS")
  endif ()
endif ()
