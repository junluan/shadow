set(Protobuf_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/protobuf CACHE PATH "Folder contains Google Protobuf")

set(Protobuf_DIR ${Protobuf_ROOT_DIR} /usr /usr/local)

find_path(Protobuf_INCLUDE_DIRS
          NAMES google/protobuf/message.h
          PATHS ${Protobuf_DIR}
          PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu
          NO_DEFAULT_PATH)

find_library(Protobuf_LIBRARIES
             NAMES protobuf libprotobuf
             PATHS ${Protobuf_DIR}
             PATH_SUFFIXES lib lib/x86_64-linux-gnu lib/aarch64-linux-gnu
             NO_DEFAULT_PATH)

find_program(Protoc_EXECUTABLE
             NAMES protoc
             PATHS ${Protobuf_DIR}
             PATH_SUFFIXES bin
             NO_DEFAULT_PATH)

set(__looked_for Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protoc_EXECUTABLE)

mark_as_advanced(Protobuf_ROOT_DIR ${__looked_for})
unset(Protobuf_DIR)

if (Protobuf_INCLUDE_DIRS)
  parse_header(${Protobuf_INCLUDE_DIRS}/google/protobuf/stubs/common.h
               GOOGLE_PROTOBUF_VERSION)
  if (GOOGLE_PROTOBUF_VERSION)
    math(EXPR PROTOBUF_MAJOR_VERSION "${GOOGLE_PROTOBUF_VERSION} / 1000000")
    math(EXPR PROTOBUF_MINOR_VERSION "${GOOGLE_PROTOBUF_VERSION} / 1000 % 1000")
    math(EXPR PROTOBUF_SUBMINOR_VERSION "${GOOGLE_PROTOBUF_VERSION} % 1000")
    set(Protobuf_VERSION "${PROTOBUF_MAJOR_VERSION}.${PROTOBUF_MINOR_VERSION}.${PROTOBUF_SUBMINOR_VERSION}")
  else ()
    set(Protobuf_VERSION "?")
  endif ()
endif ()

if (Protoc_EXECUTABLE)
  execute_process(COMMAND ${Protoc_EXECUTABLE} --version
                  OUTPUT_VARIABLE OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if ("${OUTPUT}" MATCHES "libprotoc ([0-9.]+)")
    set(Protoc_VERSION "${CMAKE_MATCH_1}")
  else ()
    set(Protoc_VERSION "?")
  endif ()
endif ()

if (NOT ("${Protoc_VERSION}" VERSION_EQUAL "${Protobuf_VERSION}"))
  message(FATAL_ERROR "Protobuf compiler version ${Protoc_VERSION} doesn't match library version ${Protobuf_VERSION}")
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Protobuf
                                  REQUIRED_VARS ${__looked_for}
                                  VERSION_VAR Protobuf_VERSION)
