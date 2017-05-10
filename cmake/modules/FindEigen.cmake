include(FindPackageHandleStandardArgs)

set(Eigen_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/eigen3 CACHE PATH "Folder contains Eigen")

set(Eigen_DIR ${Eigen_ROOT_DIR} /usr /usr/local)

find_path(Eigen_INCLUDE_DIRS
          NAMES Eigen/Eigen
          PATHS ${Eigen_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "Eigen include header"
          NO_DEFAULT_PATH)

find_package_handle_standard_args(Eigen DEFAULT_MSG Eigen_INCLUDE_DIRS)

if (Eigen_FOUND)
  file(READ ${Eigen_INCLUDE_DIRS}/Eigen/src/Core/util/Macros.h Eigen_HEADER_CONTENTS)
  string(REGEX MATCH "define EIGEN_WORLD_VERSION * +([0-9]+)"
         Eigen_VERSION_WORLD "${Eigen_HEADER_CONTENTS}")
  string(REGEX REPLACE "define EIGEN_WORLD_VERSION * +([0-9]+)" "\\1"
         Eigen_VERSION_WORLD "${Eigen_VERSION_WORLD}")
  string(REGEX MATCH "define EIGEN_MAJOR_VERSION * +([0-9]+)"
         Eigen_VERSION_MAJOR "${Eigen_HEADER_CONTENTS}")
  string(REGEX REPLACE "define EIGEN_MAJOR_VERSION * +([0-9]+)" "\\1"
         Eigen_VERSION_MAJOR "${Eigen_VERSION_MAJOR}")
  string(REGEX MATCH "define EIGEN_MINOR_VERSION * +([0-9]+)"
         Eigen_VERSION_MINOR "${Eigen_HEADER_CONTENTS}")
  string(REGEX REPLACE "define EIGEN_MINOR_VERSION * +([0-9]+)" "\\1"
         Eigen_VERSION_MINOR "${Eigen_VERSION_MINOR}")
  if (NOT Eigen_VERSION_WORLD)
    set(Eigen_VERSION "?")
  else ()
    set(Eigen_VERSION "${Eigen_VERSION_WORLD}.${Eigen_VERSION_MAJOR}.${Eigen_VERSION_MINOR}")
  endif ()
  if (NOT Eigen_FIND_QUIETLY)
    message(STATUS "Found Eigen: ${Eigen_INCLUDE_DIRS} (found version ${Eigen_VERSION})")
  endif ()
  mark_as_advanced(Eigen_ROOT_DIR Eigen_INCLUDE_DIRS)
else ()
  if (Eigen_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Eigen")
  endif ()
endif ()
