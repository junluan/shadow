include(FindPackageHandleStandardArgs)

set(gRPC_USE_STATIC_LIBS ON)

set(gRPC_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/grpc CACHE PATH "Folder contains gRPC")

set(gRPC_DIR ${gRPC_ROOT_DIR} /usr /usr/local)

find_program(gRPC_CPP_PLUGIN
             NAMES grpc_cpp_plugin
             PATHS ${gRPC_DIR}
             PATH_SUFFIXES bin
             DOC "gRPC cpp plugin"
             NO_DEFAULT_PATH)

find_path(gRPC_INCLUDE_DIRS
          NAMES grpc++/grpc++.h
          PATHS ${gRPC_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "gRPC include header grpc++.h"
          NO_DEFAULT_PATH)

set(__looked_for gRPC_CPP_PLUGIN gRPC_INCLUDE_DIRS)

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
               PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
               DOC "The path to gRPC ${__grpc_lib} library"
               NO_DEFAULT_PATH)
  mark_as_advanced(${__grpc_lib_upper}_LIBRARY)
  list(APPEND __looked_for ${__grpc_lib_upper}_LIBRARY)
  list(APPEND gRPC_LIBRARIES ${${__grpc_lib_upper}_LIBRARY})
endforeach ()

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
