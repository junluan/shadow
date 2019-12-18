include(FindPackageHandleStandardArgs)

set(MKL_USE_SINGLE_DYNAMIC_LIBRARY OFF)
set(MKL_USE_STATIC_LIBS OFF)
set(MKL_MULTI_THREADED OFF)

set(INTEL_ROOT_DIR /opt/intel CACHE PATH "Folder contains intel libs")

find_path(MKL_ROOT_DIR
          NAMES include/mkl.h
          PATHS ${INTEL_ROOT_DIR}/mkl
          NO_DEFAULT_PATH)

find_path(MKL_INCLUDE_DIRS
          NAMES mkl.h
          PATHS ${MKL_ROOT_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          NO_DEFAULT_PATH)

set(__looked_for MKL_ROOT_DIR MKL_INCLUDE_DIRS)

set(__mkl_libs "")
if (MKL_USE_SINGLE_DYNAMIC_LIBRARY)
  list(APPEND __mkl_libs rt)
else ()
  if (MKL_MULTI_THREADED)
    list(APPEND __mkl_libs intel_thread)
  else ()
     list(APPEND __mkl_libs sequential)
  endif ()
  list(APPEND __mkl_libs core intel_lp64)
endif ()

foreach (__lib ${__mkl_libs})
  set(__mkl_lib "mkl_${__lib}")
  string(TOUPPER ${__mkl_lib} __mkl_lib_upper)
  if (MKL_USE_STATIC_LIBS)
    set(__mkl_lib "lib${__mkl_lib}.a")
  endif ()
  find_library(${__mkl_lib_upper}_LIBRARY
               NAMES ${__mkl_lib}
               PATHS ${MKL_ROOT_DIR}
               PATH_SUFFIXES lib lib/intel64
               NO_DEFAULT_PATH)
  list(APPEND __looked_for ${__mkl_lib_upper}_LIBRARY)
  list(APPEND MKL_LIBRARIES ${${__mkl_lib_upper}_LIBRARY})
endforeach ()

if (MKL_USE_STATIC_LIBS)
  set(MKL_LIBRARIES "-Wl,--start-group;${MKL_LIBRARIES};-Wl,--end-group")
else ()
  set(MKL_LIBRARIES "-Wl,--no-as-needed;${MKL_LIBRARIES}")
endif ()

if (NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY)
  if (MKL_USE_STATIC_LIBS)
    set(__iomp5_libs iomp5 libiomp5mt.lib)
  else ()
    set(__iomp5_libs iomp5 libiomp5md.lib)
  endif ()
  find_library(MKL_RTL_LIBRARY ${__iomp5_libs}
               PATHS ${INTEL_ROOT_DIR} ${INTEL_ROOT_DIR}/compiler
               PATH_SUFFIXES lib lib/intel64
               NO_DEFAULT_PATH)
  list(APPEND __looked_for MKL_RTL_LIBRARY)
  list(APPEND MKL_LIBRARIES ${MKL_RTL_LIBRARY})
endif ()

find_package_handle_standard_args(MKL DEFAULT_MSG ${__looked_for})

if (MKL_FOUND)
  parse_header(${MKL_INCLUDE_DIRS}/mkl_version.h
               INTEL_MKL_VERSION)
  if (NOT INTEL_MKL_VERSION)
    set(MKL_VERSION "?")
  else ()
    set(MKL_VERSION ${INTEL_MKL_VERSION})
  endif ()
  if (NOT MKL_FIND_QUIETLY)
    message(STATUS "Found MKL: ${MKL_INCLUDE_DIRS}, ${MKL_LIBRARIES} (found version ${MKL_VERSION})")
  endif ()
  mark_as_advanced(INTEL_ROOT_DIR ${__looked_for})
else ()
  if (MKL_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find MKL")
  endif ()
endif ()
