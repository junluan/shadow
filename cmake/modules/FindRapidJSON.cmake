include(FindPackageHandleStandardArgs)

set(RapidJSON_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/rapidjson CACHE PATH "Folder contains RapidJSON")

set(RapidJSON_DIR ${RapidJSON_ROOT_DIR} /usr /usr/local)

find_path(RapidJSON_INCLUDE_DIRS
          NAMES document.h
          PATHS ${RapidJSON_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64 include/rapidjson
          DOC "RapidJSON include header"
          NO_DEFAULT_PATH)

find_package_handle_standard_args(RapidJSON DEFAULT_MSG RapidJSON_INCLUDE_DIRS)

if (RapidJSON_FOUND)
  parse_header(${RapidJSON_INCLUDE_DIRS}/rapidjson.h
               RAPIDJSON_MAJOR_VERSION RAPIDJSON_MINOR_VERSION RAPIDJSON_PATCH_VERSION)
  if (NOT RAPIDJSON_MAJOR_VERSION)
    set(RapidJSON_VERSION "?")
  else ()
    set(RapidJSON_VERSION "${RAPIDJSON_MAJOR_VERSION}.${RAPIDJSON_MINOR_VERSION}.${RAPIDJSON_PATCH_VERSION}")
  endif ()
  if (NOT RapidJSON_FIND_QUIETLY)
    message(STATUS "Found RapidJSON: ${RapidJSON_INCLUDE_DIRS} (found version ${RapidJSON_VERSION})")
  endif ()
  mark_as_advanced(RapidJSON_ROOT_DIR RapidJSON_INCLUDE_DIRS)
else ()
  if (RapidJSON_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find RapidJSON")
  endif ()
endif ()
