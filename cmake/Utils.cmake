####################################################################
# Command alias for debugging messages
# Usage:
#   dmsg(<message>)
function (dmsg)
  message(STATUS ${ARGN})
endfunction ()

####################################################################
# Removes duplicates from list(s)
# Usage:
#   shadow_list_unique(<list_variable> [<list_variable>] [...])
macro (shadow_list_unique)
  foreach (__lst ${ARGN})
    if (${__lst})
      list(REMOVE_DUPLICATES ${__lst})
    endif ()
  endforeach ()
endmacro ()

####################################################################
# Clears variables from list
# Usage:
#   shadow_clear_vars(<variables_list>)
macro (shadow_clear_vars)
  foreach (_var ${ARGN})
    unset(${_var})
  endforeach ()
endmacro ()

####################################################################
# Removes duplicates from string
# Usage:
#   shadow_string_unique(<string_variable>)
function (shadow_string_unique __string)
  if (${__string})
    set(__list ${${__string}})
    separate_arguments(__list)
    list(REMOVE_DUPLICATES __list)
    foreach (__e ${__list})
      set(__str "${__str} ${__e}")
    endforeach ()
    set(${__string} ${__str} PARENT_SCOPE)
  endif ()
endfunction ()

####################################################################
# Prints list element per line
# Usage:
#   shadow_print_list(<list>)
function (shadow_print_list)
  foreach (e ${ARGN})
    message(STATUS ${e})
  endforeach ()
endfunction ()

####################################################################
# Reads set of version defines from the header file
# Usage:
#   shadow_parse_header(<file> <define1> <define2> <define3> ..)
function (shadow_parse_header FILENAME)
  set(vars_regex "")
  foreach (name ${ARGN})
    if (vars_regex)
      set(vars_regex "${vars_regex}|${name}")
    else ()
      set(vars_regex "${name}")
    endif ()
  endforeach ()
  set(HEADER_CONTENTS "")
  if (EXISTS ${FILENAME})
    file(STRINGS ${FILENAME} HEADER_CONTENTS REGEX "#define[ \t]+(${vars_regex})[ \t]+[0-9]+")
  endif ()
  foreach (name ${ARGN})
    set(num "")
    if (HEADER_CONTENTS MATCHES ".+[ \t]${name}[ \t]+([0-9]+).*")
      string(REGEX REPLACE ".+[ \t]${name}[ \t]+([0-9]+).*" "\\1" num "${HEADER_CONTENTS}")
    endif ()
    set(${name} ${num} PARENT_SCOPE)
  endforeach ()
endfunction ()

####################################################################
# Reads set of version defines from the header file
# Usage:
#   shadow_parse_header_single_define(<file> <define> <regex>)
function (shadow_parse_header_single_define FILENAME VARNAME REGEX)
  set(HEADER_CONTENTS "")
  if (EXISTS ${FILENAME})
    file(STRINGS ${FILENAME} HEADER_CONTENTS REGEX "#define[ \t]+${VARNAME}[ \t]+\".+\"")
  endif ()
  set(version "")
  if (HEADER_CONTENTS MATCHES ".*(${REGEX}).*")
    string(REGEX REPLACE ".*(${REGEX}).*" "\\1" version "${HEADER_CONTENTS}")
  endif ()
  set(${VARNAME} ${version} PARENT_SCOPE)
endfunction ()

####################################################################
# Add whole archive when build static library
# Usage:
#   shadow_add_whole_archive_flag(<lib> <output_var>)
function (shadow_add_whole_archive_flag lib output_var)
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    if (MSVC_VERSION GREATER 1900)
      set(${output_var} -WHOLEARCHIVE:$<TARGET_FILE:${lib}> PARENT_SCOPE)
    else ()
      message(WARNING "MSVC version is ${MSVC_VERSION}, /WHOLEARCHIVE flag cannot be set")
      set(${output_var} ${lib} PARENT_SCOPE)
    endif ()
  elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    #set(${output_var} -Wl,-force_load,$<TARGET_FILE:${lib}> PARENT_SCOPE)
    set(${output_var} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive PARENT_SCOPE)
  else ()
    set(${output_var} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive PARENT_SCOPE)
  endif ()
endfunction ()

####################################################################
# Find current os platform and architecture
# Usage:
#   shadow_find_os_arch(<output_var>)
function (shadow_find_os_arch platform_var arch_var)
  set(${platform_var})
  set(${arch_var})
  if (MSVC)
    set(${platform_var} windows PARENT_SCOPE)
    set(${arch_var} x86_64 PARENT_SCOPE)
  elseif (ANDROID)
    set(${platform_var} android PARENT_SCOPE)
    set(${arch_var} ${ANDROID_ABI} PARENT_SCOPE)
  elseif (APPLE)
    set(${platform_var} darwin PARENT_SCOPE)
    set(${arch_var} x86_64 PARENT_SCOPE)
  elseif (UNIX AND NOT APPLE)
    set(${platform_var} linux PARENT_SCOPE)
    set(${arch_var} x86_64 PARENT_SCOPE)
  endif ()
endfunction ()
