set(Eigen_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/eigen3 CACHE PATH "Folder contains Eigen")

set(Eigen_DIR ${Eigen_ROOT_DIR} /usr /usr/local)

find_path(Eigen_INCLUDE_DIRS
          NAMES Eigen/Eigen
          PATHS ${Eigen_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64 include/eigen3
          NO_DEFAULT_PATH)

mark_as_advanced(Eigen_ROOT_DIR Eigen_INCLUDE_DIRS)
unset(Eigen_DIR)

if (Eigen_INCLUDE_DIRS)
  parse_header(${Eigen_INCLUDE_DIRS}/Eigen/src/Core/util/Macros.h
               EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
  if (EIGEN_WORLD_VERSION)
    set(Eigen_VERSION "${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}")
  else ()
    set(Eigen_VERSION "?")
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen
                                  REQUIRED_VARS Eigen_INCLUDE_DIRS
                                  VERSION_VAR Eigen_VERSION)
