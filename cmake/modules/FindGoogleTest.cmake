include(FindPackageHandleStandardArgs)

set(GoogleTest_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/googletest CACHE PATH "Folder contains googletest")

set(GoogleTest_DIR ${GoogleTest_ROOT_DIR}/build /usr /usr/local)

find_path(GoogleTest_INCLUDE_DIRS
          NAMES gtest/gtest.h
          PATHS ${GoogleTest_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          NO_DEFAULT_PATH)

find_library(GoogleTest_LIBRARIES
             NAMES gtest
             PATHS ${GoogleTest_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             NO_DEFAULT_PATH)

find_package_handle_standard_args(GoogleTest DEFAULT_MSG GoogleTest_INCLUDE_DIRS GoogleTest_LIBRARIES)

if (GoogleTest_FOUND)
  if (NOT GoogleTest_FIND_QUIETLY)
    message(STATUS "Found googletest: ${GoogleTest_INCLUDE_DIRS}, ${GoogleTest_LIBRARIES}")
  endif ()
  mark_as_advanced(GoogleTest_ROOT_DIR GoogleTest_INCLUDE_DIRS GoogleTest_LIBRARIES)
else ()
  if (GoogleTest_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find googletest")
  endif ()
endif ()
