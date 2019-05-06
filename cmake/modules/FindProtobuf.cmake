include(FindPackageHandleStandardArgs)

set(Protobuf_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/protobuf CACHE PATH "Folder contains Google Protobuf")

set(Protobuf_DIR ${Protobuf_ROOT_DIR} /usr /usr/local)

find_path(Protobuf_INCLUDE_DIRS
          NAMES google/protobuf/message.h
          PATHS ${Protobuf_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          NO_DEFAULT_PATH)

find_library(Protobuf_LIBRARIES
             NAMES protobuf libprotobuf
             PATHS ${Protobuf_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
             NO_DEFAULT_PATH)

find_program(Protoc_EXECUTABLE
             NAMES protoc
             PATHS ${Protobuf_DIR}
             PATH_SUFFIXES bin
             NO_DEFAULT_PATH)

find_package_handle_standard_args(Protobuf DEFAULT_MSG Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protoc_EXECUTABLE)

if (Protobuf_FOUND)
  parse_header(${Protobuf_INCLUDE_DIRS}/google/protobuf/stubs/common.h
               GOOGLE_PROTOBUF_VERSION)
  math(EXPR PROTOBUF_MAJOR_VERSION "${GOOGLE_PROTOBUF_VERSION} / 1000000")
  math(EXPR PROTOBUF_MINOR_VERSION "${GOOGLE_PROTOBUF_VERSION} / 1000 % 1000")
  math(EXPR PROTOBUF_SUBMINOR_VERSION "${GOOGLE_PROTOBUF_VERSION} % 1000")
  set(Protobuf_VERSION "${PROTOBUF_MAJOR_VERSION}.${PROTOBUF_MINOR_VERSION}.${PROTOBUF_SUBMINOR_VERSION}")
  execute_process(COMMAND ${Protoc_EXECUTABLE} --version OUTPUT_VARIABLE Protoc_VERSION)
  if ("${Protoc_VERSION}" MATCHES "libprotoc ([0-9.]+)")
    set(Protoc_VERSION "${CMAKE_MATCH_1}")
  endif ()
  if (NOT "${Protoc_VERSION}" VERSION_EQUAL "${Protobuf_VERSION}")
    message(FATAL_ERROR "Protobuf compiler version ${Protoc_VERSION} doesn't match library version ${Protobuf_VERSION}")
  endif ()
  if (NOT Protobuf_FIND_QUIETLY)
    message(STATUS "Found Protobuf: ${Protobuf_INCLUDE_DIRS}, ${Protobuf_LIBRARIES} (found version ${Protobuf_VERSION})")
    message(STATUS "Found Protoc: ${Protoc_EXECUTABLE} (found version ${Protoc_VERSION})")
  endif ()
  mark_as_advanced(Protobuf_ROOT_DIR Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protoc_EXECUTABLE)
else ()
  if (Protobuf_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Protobuf")
  endif ()
endif ()
