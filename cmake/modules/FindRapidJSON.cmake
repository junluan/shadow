set(RapidJSON_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/rapidjson CACHE PATH "Folder contains RapidJSON")

set(RapidJSON_DIR ${RapidJSON_ROOT_DIR} /usr /usr/local)

find_path(RapidJSON_INCLUDE_DIRS
          NAMES document.h
          PATHS ${RapidJSON_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64 include/rapidjson
          NO_DEFAULT_PATH)

mark_as_advanced(RapidJSON_ROOT_DIR RapidJSON_INCLUDE_DIRS)
unset(RapidJSON_DIR)

if (RapidJSON_INCLUDE_DIRS)
  parse_header(${RapidJSON_INCLUDE_DIRS}/rapidjson.h
               RAPIDJSON_MAJOR_VERSION RAPIDJSON_MINOR_VERSION RAPIDJSON_PATCH_VERSION)
  if (RAPIDJSON_MAJOR_VERSION)
    set(RapidJSON_VERSION "${RAPIDJSON_MAJOR_VERSION}.${RAPIDJSON_MINOR_VERSION}.${RAPIDJSON_PATCH_VERSION}")
  else ()
    set(RapidJSON_VERSION "?")
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RapidJSON
                                  REQUIRED_VARS RapidJSON_INCLUDE_DIRS
                                  VERSION_VAR RapidJSON_VERSION)
