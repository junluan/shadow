set(OpenCL_PATHS
    /usr
    /usr/local
    /usr/local/cuda)

find_path(OpenCL_INCLUDE_DIRS
          NAMES OpenCL/cl.h CL/cl.h
          PATHS ${OpenCL_PATHS}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "OpenCL include header OpenCL/cl.h or CL/cl.h")

find_library(OpenCL_LIBRARIES
             NAMES OpenCL
             PATHS ${OPENCL_PATHS}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "OpenCL library")

set(OpenCL_FOUND ON)

if (NOT OpenCL_INCLUDE_DIRS)
  set(OpenCL_FOUND OFF)
  message(STATUS "Could not find OpenCL include. Turning OpenCL_FOUND off")
endif ()

if (NOT OpenCL_LIBRARIES)
  set(OpenCL_FOUND OFF)
  message(STATUS "Could not find OpenCL lib. Turning OpenCL_FOUND off")
endif ()

if (OpenCL_FOUND)
  if (APPLE)
    set(CL_HEADER_FILE "${OpenCL_INCLUDE_DIRS}/OpenCL/cl.h")
  else ()
    set(CL_HEADER_FILE "${OpenCL_INCLUDE_DIRS}/CL/cl.h")
  endif ()
  check_symbol_exists(CL_VERSION_2_0 ${CL_HEADER_FILE} HAVE_CL_2_0)
  if (HAVE_CL_2_0)
    set(OpenCL_VERSION_STRING "2.0")
  else ()
    check_symbol_exists(CL_VERSION_1_2 ${CL_HEADER_FILE} HAVE_CL_1_2)
    if (HAVE_CL_1_2)
      set(OpenCL_VERSION_STRING "1.2")
    else ()
      set(OpenCL_VERSION_STRING "0.0")
    endif ()
  endif ()
  if (NOT OpenCL_FIND_QUIETLY)
    message(STATUS "Found OpenCL include: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "Found OpenCL libraries: ${OpenCL_LIBRARIES}")
  endif ()
  mark_as_advanced(OpenCL_INCLUDE_DIRS OpenCL_LIBRARIES OpenCL)
else ()
  if (OpenCL_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find OpenCL")
  endif ()
endif ()
