function (detect_CUDNN)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  set(CUDNN_PATHS
      ${CUDNN_ROOT}
      $ENV{CUDNN_ROOT}
      ${CUDA_TOOLKIT_ROOT_DIR})

  find_path(CUDNN_INCLUDE_DIRS
            NAMES cudnn.h
            PATHS ${CUDNN_PATHS}
            PATH_SUFFIXES include include/x86_64 include/x64
            DOC "CUDNN include header cudnn.h" )

  find_library(CUDNN_LIBRARIES
               NAMES libcudnn.so
               PATHS ${CUDNN_PATHS}
               PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
               DOC "CUDNN library")
  
  if (CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    file(READ ${CUDNN_INCLUDE_DIRS}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

    # CUDNN v3 and beyond
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    if (NOT CUDNN_VERSION_MAJOR)
      set(CUDNN_VERSION "???")
    else ()
      set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif ()

    string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 CUDNNVersionIncompatible)
    if (CUDNNVersionIncompatible)
      message(FATAL_ERROR "CUDNN version >3 is required.")
    endif ()

    set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES CUDNN_VERSION CUDNN)
  endif ()
endfunction ()
