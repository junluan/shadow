include(FindPackageHandleStandardArgs)

set(clBLAS_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/clblas CACHE PATH "Folder contains clBLAS")

set(clBLAS_DIR ${clBLAS_ROOT_DIR} /usr /usr/local)

set(clBLAS_PLATFORM)
set(clBLAS_ARC)
if (MSVC)
  set(clBLAS_PLATFORM windows)
  set(clBLAS_ARC x86_64)
elseif (ANDROID)
  set(clBLAS_PLATFORM android)
  set(clBLAS_ARC ${ANDROID_ABI})
elseif (APPLE)
  set(clBLAS_PLATFORM darwin)
  set(clBLAS_ARC x86_64)
elseif (Linux)
  set(clBLAS_PLATFORM linux)
  set(clBLAS_ARC x86_64)
endif ()

find_path(clBLAS_INCLUDE_DIRS
          NAMES clBLAS.h
          PATHS ${clBLAS_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "clBLAS include header clBLAS.h"
          NO_DEFAULT_PATH)

find_library(clBLAS_LIBRARIES
             NAMES clBLAS
             PATHS ${clBLAS_DIR}
             PATH_SUFFIXES lib lib/${clBLAS_PLATFORM}/${clBLAS_ARC} lib64 lib/x86_64 lib/x64 lib/x86
             DOC "clBLAS library"
             NO_DEFAULT_PATH)

find_package_handle_standard_args(clBLAS DEFAULT_MSG clBLAS_INCLUDE_DIRS clBLAS_LIBRARIES)

if (clBLAS_FOUND)
  shadow_parse_header(${clBLAS_INCLUDE_DIRS}/clBLAS.version.h
                      clblasVersionMajor clblasVersionMinor clblasVersionPatch)
  if (NOT clblasVersionMajor)
    set(clBLAS_VERSION "?")
  else ()
    set(clBLAS_VERSION "${clblasVersionMajor}.${clblasVersionMinor}.${clblasVersionPatch}")
  endif ()
  if (NOT clBLAS_FIND_QUIETLY)
    message(STATUS "Found clBLAS: ${clBLAS_INCLUDE_DIRS}, ${clBLAS_LIBRARIES} (found version ${clBLAS_VERSION})")
  endif ()
  mark_as_advanced(clBLAS_ROOT_DIR clBLAS_INCLUDE_DIRS clBLAS_LIBRARIES)
else ()
  if (clBLAS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find clBLAS")
  endif ()
endif ()
