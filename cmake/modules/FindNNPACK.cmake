set(NNPACK_ROOT_DIR
    ./third_party/nnpack
    /usr
    /usr/local)

find_path(NNPACK_INCLUDE_DIRS
          NAMES nnpack.h
          PATHS ${NNPACK_ROOT_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "NNPACK include header nnpack.h"
          NO_DEFAULT_PATH)

find_library(NNPACK_LIBRARY
             NAMES nnpack
             PATHS ${NNPACK_ROOT_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "NNPACK library"
             NO_DEFAULT_PATH)

find_library(PTHREADPOOL_LIBRARY
             NAMES pthreadpool
             PATHS ${NNPACK_ROOT_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "NNPACK library"
             NO_DEFAULT_PATH)

set(NNPACK_FOUND ON)

if (NOT NNPACK_INCLUDE_DIRS)
  set(NNPACK_FOUND OFF)
  if (NOT NNPACK_FIND_QUIETLY)
    message(STATUS "Could not find NNPACK include. Turning NNPACK_FOUND off")
  endif ()
endif ()

if (NOT NNPACK_LIBRARY OR NOT PTHREADPOOL_LIBRARY)
  set(NNPACK_FOUND OFF)
  if (NOT NNPACK_FIND_QUIETLY)
    message(STATUS "Could not find NNPACK lib. Turning NNPACK_FOUND off")
  endif ()
endif ()

if (NNPACK_FOUND)
  set(NNPACK_LIBRARIES "${NNPACK_LIBRARY};${PTHREADPOOL_LIBRARY}")
  if (NOT NNPACK_FIND_QUIETLY)
    message(STATUS "Found NNPACK include: ${NNPACK_INCLUDE_DIRS}")
    message(STATUS "Found NNPACK libraries: ${NNPACK_LIBRARIES}")
  endif ()
  mark_as_advanced(NNPACK_INCLUDE_DIRS NNPACK_LIBRARIES NNPACK)
else ()
  if (NNPACK_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find NNPACK")
  endif ()
endif ()
