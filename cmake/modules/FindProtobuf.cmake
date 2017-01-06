set(Protobuf_PATHS
    ./external/protobuf
    /usr
    /usr/local)

find_path(Protobuf_INCLUDE_DIRS
          NAMES google/protobuf/message.h
          PATHS ${Protobuf_PATHS}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "Protobuf include"
          NO_DEFAULT_PATH)

find_library(Protobuf_LIBRARIES
             NAMES protobuf
             PATHS ${Protobuf_PATHS}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
             DOC "Protobuf library"
             NO_DEFAULT_PATH)

find_program(Protobuf_PROTOC_EXECUTABLE
             NAMES protoc
             PATHS ${Protobuf_PATHS}
             PATH_SUFFIXES bin
             DOC "Protobuf protoc"
             NO_DEFAULT_PATH)

set(Protobuf_FOUND ON)

if (NOT Protobuf_INCLUDE_DIRS)
  set(Protobuf_FOUND OFF)
  if (NOT Protobuf_FIND_QUIETLY)
    message(STATUS "Could not find Protobuf include. Turning Protobuf_FOUND off")
  endif ()
endif ()

if (NOT Protobuf_LIBRARIES)
  set(Protobuf_FOUND OFF)
  if (NOT Protobuf_FIND_QUIETLY)
    message(STATUS "Could not find Protobuf lib. Turning Protobuf_FOUND off")
  endif ()
endif ()

if (NOT Protobuf_PROTOC_EXECUTABLE)
  set(Protobuf_FOUND OFF)
  if (NOT Protobuf_FIND_QUIETLY)
    message(STATUS "Could not find Protobuf protoc. Turning Protobuf_FOUND off")
  endif ()
endif ()

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
  execute_process(COMMAND ${Protobuf_PROTOC_EXECUTABLE} --version
                  OUTPUT_VARIABLE PROTOBUF_PROTOC_EXECUTABLE_VERSION)
  if ("${PROTOBUF_PROTOC_EXECUTABLE_VERSION}" MATCHES "libprotoc ([0-9.]+)")
    set(PROTOBUF_PROTOC_EXECUTABLE_VERSION "${CMAKE_MATCH_1}")
  endif ()
  if (NOT "${PROTOBUF_PROTOC_EXECUTABLE_VERSION}" VERSION_EQUAL "${Protobuf_VERSION}")
    message(FATAL_ERROR "Protobuf compiler version ${PROTOBUF_PROTOC_EXECUTABLE_VERSION}"
            " doesn't match library version ${Protobuf_VERSION}")
  endif ()
  if (NOT Protobuf_FIND_QUIETLY)
    message(STATUS "Found Protobuf include: ${Protobuf_INCLUDE_DIRS}")
    message(STATUS "Found Protobuf libraries: ${Protobuf_LIBRARIES}")
    message(STATUS "Found Protobuf protoc: ${Protobuf_PROTOC_EXECUTABLE}")
  endif ()
  mark_as_advanced(Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protobuf_PROTOC_EXECUTABLE Protobuf)
else ()
  if (Protobuf_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Protobuf")
  endif ()
endif ()
