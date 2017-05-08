include(FindPackageHandleStandardArgs)

set(clBLAS_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/clblas CACHE PATH "Folder contains clBLAS")

set(clBLAS_DIR ${clBLAS_ROOT_DIR} /usr /usr/local)

find_path(clBLAS_INCLUDE_DIRS
          NAMES clBLAS.h
          PATHS ${clBLAS_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "clBLAS include header clBLAS.h"
          NO_DEFAULT_PATH)

find_library(clBLAS_LIBRARIES
             NAMES clBLAS
             PATHS ${clBLAS_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "clBLAS library"
             NO_DEFAULT_PATH)

find_package_handle_standard_args(clBLAS DEFAULT_MSG clBLAS_INCLUDE_DIRS clBLAS_LIBRARIES)

if (clBLAS_FOUND)
  file(READ ${clBLAS_INCLUDE_DIRS}/clBLAS.version.h clBLAS_HEADER_CONTENTS)
  string(REGEX MATCH "define clblasVersionMajor * +([0-9]+)"
         clBLAS_VERSION_MAJOR "${clBLAS_HEADER_CONTENTS}")
  string(REGEX REPLACE "define clblasVersionMajor * +([0-9]+)" "\\1"
         clBLAS_VERSION_MAJOR "${clBLAS_VERSION_MAJOR}")
  string(REGEX MATCH "define clblasVersionMinor * +([0-9]+)"
         clBLAS_VERSION_MINOR "${clBLAS_HEADER_CONTENTS}")
  string(REGEX REPLACE "define clblasVersionMinor * +([0-9]+)" "\\1"
         clBLAS_VERSION_MINOR "${clBLAS_VERSION_MINOR}")
  string(REGEX MATCH "define clblasVersionPatch * +([0-9]+)"
         clBLAS_VERSION_PATCH "${clBLAS_HEADER_CONTENTS}")
  string(REGEX REPLACE "define clblasVersionPatch * +([0-9]+)" "\\1"
         clBLAS_VERSION_PATCH "${clBLAS_VERSION_PATCH}")
  if (NOT clBLAS_VERSION_MAJOR)
    set(clBLAS_VERSION "?")
  else ()
    set(clBLAS_VERSION "${clBLAS_VERSION_MAJOR}.${clBLAS_VERSION_MINOR}.${clBLAS_VERSION_PATCH}")
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
