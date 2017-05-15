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
