include(FindPackageHandleStandardArgs)

set(Eigen_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/eigen3 CACHE PATH "Folder contains Eigen")

set(Eigen_DIR ${Eigen_ROOT_DIR} /usr /usr/local)

find_path(Eigen_INCLUDE_DIRS
          NAMES Eigen/Eigen
          PATHS ${Eigen_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64 include/eigen3
          DOC "Eigen include header"
          NO_DEFAULT_PATH)

find_package_handle_standard_args(Eigen DEFAULT_MSG Eigen_INCLUDE_DIRS)

if (Eigen_FOUND)
  parse_header(${Eigen_INCLUDE_DIRS}/Eigen/src/Core/util/Macros.h
               EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
  if (NOT EIGEN_WORLD_VERSION)
    set(Eigen_VERSION "?")
  else ()
    set(Eigen_VERSION "${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}")
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
