include(FindPackageHandleStandardArgs)

set(Protobuf_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/protobuf CACHE PATH "Folder contains Google Protobuf")

set(Protobuf_DIR ${Protobuf_ROOT_DIR} /usr /usr/local)

find_program(Protoc_EXECUTABLE
             NAMES protoc
             PATHS ${Protobuf_DIR}
             PATH_SUFFIXES bin
             DOC "Protobuf protoc"
             NO_DEFAULT_PATH)

find_path(Protobuf_INCLUDE_DIRS
          NAMES google/protobuf/message.h
          PATHS ${Protobuf_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "Protobuf include"
          NO_DEFAULT_PATH)

if (NOT MSVC)
  find_library(Protobuf_LIBRARIES
               NAMES protobuf libprotobuf
               PATHS ${Protobuf_DIR}
               PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
               DOC "Protobuf library"
               NO_DEFAULT_PATH)
else ()
  find_library(Protobuf_LIBRARIES_RELEASE
               NAMES libprotobuf
               PATHS ${Protobuf_DIR}
               PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
               DOC "Protobuf library"
               NO_DEFAULT_PATH)
  find_library(Protobuf_LIBRARIES_DEBUG
               NAMES libprotobufd
               PATHS ${Protobuf_DIR}
               PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
               DOC "Protobuf library"
               NO_DEFAULT_PATH)

  set(Protobuf_LIBRARIES optimized ${Protobuf_LIBRARIES_RELEASE} debug ${Protobuf_LIBRARIES_DEBUG})
endif ()

find_package_handle_standard_args(Protobuf DEFAULT_MSG Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protoc_EXECUTABLE)

if (Protobuf_FOUND)
  set(Protobuf_VERSION "")
  set(Protobuf_LIB_VERSION "")
  set(PROTOBUF_COMMON_FILE ${Protobuf_INCLUDE_DIRS}/google/protobuf/stubs/common.h)
  file(STRINGS ${PROTOBUF_COMMON_FILE}
       PROTOBUF_COMMON_H_CONTENTS
       REGEX "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+")
  if (PROTOBUF_COMMON_H_CONTENTS MATCHES "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+([0-9]+)")
    set(Protobuf_LIB_VERSION "${CMAKE_MATCH_1}")
  endif ()
  unset(PROTOBUF_COMMON_H_CONTENTS)
  math(EXPR PROTOBUF_MAJOR_VERSION "${Protobuf_LIB_VERSION} / 1000000")
  math(EXPR PROTOBUF_MINOR_VERSION "${Protobuf_LIB_VERSION} / 1000 % 1000")
  math(EXPR PROTOBUF_SUBMINOR_VERSION "${Protobuf_LIB_VERSION} % 1000")
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
  mark_as_advanced(Protobuf_ROOT_DIR Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protobuf_LIBRARIES_RELEASE Protobuf_LIBRARIES_DEBUG Protoc_EXECUTABLE)
else ()
  if (Protobuf_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Protobuf")
  endif ()
endif ()
