include(FindPackageHandleStandardArgs)

set(NNPACK_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/nnpack CACHE PATH "Folder contains NNPACK")

set(NNPACK_DIR ${NNPACK_ROOT_DIR}/build ${NNPACK_ROOT_DIR} /usr /usr/local)

set(NNPACK_PLATFORM)
set(NNPACK_ARC)
set(NNPACK_LIBS)
if (MSVC)
  set(NNPACK_PLATFORM windows)
  set(NNPACK_ARC x86_64)
elseif (ANDROID)
  set(NNPACK_PLATFORM android)
  set(NNPACK_ARC ${ANDROID_ABI})
elseif (APPLE)
  set(NNPACK_PLATFORM darwin)
  set(NNPACK_ARC x86_64)
elseif (UNIX AND NOT APPLE)
  set(NNPACK_PLATFORM linux)
  set(NNPACK_ARC x86_64)
endif ()

find_path(NNPACK_INCLUDE_DIRS
          NAMES nnpack.h
          PATHS ${NNPACK_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "NNPACK include header nnpack.h"
          NO_DEFAULT_PATH)

find_library(NNPACK_LIBRARY
             NAMES nnpack
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib/${NNPACK_PLATFORM}/${NNPACK_ARC} lib64 lib/x86_64 lib/x64 lib/x86
             DOC "NNPACK library"
             NO_DEFAULT_PATH)

find_library(PTHREADPOOL_LIBRARY
             NAMES pthreadpool
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib/${NNPACK_PLATFORM}/${NNPACK_ARC} lib64 lib/x86_64 lib/x64 lib/x86
             DOC "NNPACK library"
             NO_DEFAULT_PATH)

find_library(CPUINFO_LIBRARY
             NAMES cpuinfo
             PATHS ${NNPACK_DIR}
             PATH_SUFFIXES lib lib/${NNPACK_PLATFORM}/${NNPACK_ARC} lib64 lib/x86_64 lib/x64 lib/x86
             DOC "NNPACK library"
             NO_DEFAULT_PATH)

set(__looked_for NNPACK_INCLUDE_DIRS NNPACK_LIBRARY PTHREADPOOL_LIBRARY CPUINFO_LIBRARY)
set(NNPACK_LIBRARIES ${NNPACK_LIBRARY} ${PTHREADPOOL_LIBRARY} ${CPUINFO_LIBRARY})

if (ANDROID)
  set(__nnpack_libs nnpack_ukernels)
  if (${NNPACK_ARC} STREQUAL "armeabi-v7a")
    list(APPEND __nnpack_libs cpufeatures)
  endif ()
  foreach (__nnpack_lib ${__nnpack_libs})
    string(TOUPPER ${__nnpack_lib} __nnpack_lib_upper)
    find_library(${__nnpack_lib_upper}_LIBRARY
                 NAMES ${__nnpack_lib}
                 PATHS ${NNPACK_DIR}
                 PATH_SUFFIXES lib lib/${NNPACK_PLATFORM}/${NNPACK_ARC} lib64 lib/x86_64 lib/x64 lib/x86
                 DOC "The path to nnpack ${__nnpack_lib} library"
                 NO_DEFAULT_PATH)
    mark_as_advanced(${__nnpack_lib_upper}_LIBRARY)
    list(APPEND __looked_for ${__nnpack_lib_upper}_LIBRARY)
    list(APPEND NNPACK_LIBRARIES ${${__nnpack_lib_upper}_LIBRARY})
  endforeach ()
endif ()

find_package_handle_standard_args(NNPACK DEFAULT_MSG ${__looked_for})

if (NNPACK_FOUND)
  if (NOT NNPACK_FIND_QUIETLY)
    message(STATUS "Found NNPACK: ${NNPACK_INCLUDE_DIRS}, ${NNPACK_LIBRARIES}")
  endif ()
  mark_as_advanced(NNPACK_ROOT_DIR NNPACK_INCLUDE_DIRS NNPACK_LIBRARY PTHREADPOOL_LIBRARY)
else ()
  if (NNPACK_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find NNPACK")
  endif ()
endif ()
