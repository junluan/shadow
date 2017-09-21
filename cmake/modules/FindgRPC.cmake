include(FindPackageHandleStandardArgs)

set(gRPC_USE_STATIC_LIBS ON)

set(gRPC_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/grpc CACHE PATH "Folder contains gRPC")

set(gRPC_DIR ${gRPC_ROOT_DIR} /usr /usr/local)

set(gRPC_PLATFORM)
set(gRPC_ARC)
if (MSVC)
  set(gRPC_PLATFORM windows)
  set(gRPC_ARC x86_64)
elseif (ANDROID)
  set(gRPC_PLATFORM android)
  set(gRPC_ARC ${ANDROID_ABI})
elseif (APPLE)
  set(gRPC_PLATFORM darwin)
  set(gRPC_ARC x86_64)
elseif (UNIX AND NOT APPLE)
  set(gRPC_PLATFORM linux)
  set(gRPC_ARC x86_64)
endif ()

find_program(gRPC_CPP_PLUGIN
             NAMES grpc_cpp_plugin
             PATHS ${gRPC_DIR}
             PATH_SUFFIXES bin bin/${gRPC_PLATFORM}/${gRPC_ARC}
             DOC "gRPC cpp plugin"
             NO_DEFAULT_PATH)

find_path(gRPC_INCLUDE_DIRS
          NAMES grpc++/grpc++.h
          PATHS ${gRPC_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "gRPC include header grpc++.h"
          NO_DEFAULT_PATH)

set(__looked_for gRPC_CPP_PLUGIN gRPC_INCLUDE_DIRS)

if (NOT MSVC)
  set(__grpc_libs grpc++_unsecure)
  if (gRPC_USE_STATIC_LIBS)
    list(APPEND __grpc_libs grpc)
  endif ()
  foreach (__grpc_lib ${__grpc_libs})
    string(TOUPPER ${__grpc_lib} __grpc_lib_upper)
    if (gRPC_USE_STATIC_LIBS)
      set(__grpc_lib "lib${__grpc_lib}.a")
    endif ()
    find_library(${__grpc_lib_upper}_LIBRARY
                 NAMES ${__grpc_lib}
                 PATHS ${gRPC_DIR}
                 PATH_SUFFIXES lib lib/${gRPC_PLATFORM}/${gRPC_ARC} lib64 lib/x86_64 lib/x64 lib/x86
                 DOC "The path to gRPC ${__grpc_lib} library"
                 NO_DEFAULT_PATH)
    mark_as_advanced(${__grpc_lib_upper}_LIBRARY)
    list(APPEND __looked_for ${__grpc_lib_upper}_LIBRARY)
    list(APPEND gRPC_LIBRARIES ${${__grpc_lib_upper}_LIBRARY})
  endforeach ()
else ()
  set(gRPC_RUNTIME)
  if (MSVC_VERSION EQUAL 1800)
    set(gRPC_RUNTIME vc120)
  elseif (MSVC_VERSION EQUAL 1900)
    set(gRPC_RUNTIME vc140)
  elseif (MSVC_VERSION EQUAL 1910)
    set(gRPC_RUNTIME vc141)
  endif ()
  set(__grpc_libs grpc++_unsecure grpc_dll)
  foreach (__grpc_lib ${__grpc_libs})
    string(TOUPPER ${__grpc_lib} __grpc_lib_upper)
    set(__grpc_lib_win "${__grpc_lib}.lib")
    find_library(${__grpc_lib_upper}_LIBRARY_RELEASE
                 NAMES ${__grpc_lib_win}
                 PATHS ${gRPC_DIR}
                 PATH_SUFFIXES lib/${gRPC_PLATFORM}/${gRPC_ARC}/${gRPC_RUNTIME}
                 DOC "The path to gRPC ${__grpc_lib_win} library"
                 NO_DEFAULT_PATH)
    set(__grpc_lib_win "${__grpc_lib}d.lib")
    find_library(${__grpc_lib_upper}_LIBRARY_DEBUG
                 NAMES ${__grpc_lib_win}
                 PATHS ${gRPC_DIR}
                 PATH_SUFFIXES lib/${gRPC_PLATFORM}/${gRPC_ARC}/${gRPC_RUNTIME}
                 DOC "The path to gRPC ${__grpc_lib_win} library"
                 NO_DEFAULT_PATH)
    mark_as_advanced(${__grpc_lib_upper}_LIBRARY_RELEASE ${__grpc_lib_upper}_LIBRARY_DEBUG)
    list(APPEND __looked_for ${__grpc_lib_upper}_LIBRARY_RELEASE ${__grpc_lib_upper}_LIBRARY_DEBUG)
    list(APPEND gRPC_LIBRARIES optimized ${${__grpc_lib_upper}_LIBRARY_RELEASE} debug ${${__grpc_lib_upper}_LIBRARY_DEBUG})
  endforeach ()
endif ()

find_package_handle_standard_args(gRPC DEFAULT_MSG ${__looked_for})

if (gRPC_FOUND)
  if (NOT gRPC_FIND_QUIETLY)
    message(STATUS "Found gRPC: ${gRPC_INCLUDE_DIRS}, ${gRPC_LIBRARIES}")
  endif ()
  mark_as_advanced(gRPC_ROOT_DIR gRPC_INCLUDE_DIRS gRPC_CPP_PLUGIN)
else ()
  if (gRPC_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find gRPC")
  endif ()
endif ()
