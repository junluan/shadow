file(GLOB config_src_in "config.hpp.in")
string(REPLACE "hpp.in" "hpp" config_src ${config_src_in})
configure_file(${config_src_in} ${config_src} @ONLY)

file(GLOB_RECURSE tmp "*.cpp" "*.hpp")
set(shadow_lib_src ${shadow_lib_src} ${tmp})

if (${USE_CUDA})
  file(GLOB_RECURSE tmp "*.cu")
  set(shadow_lib_src ${shadow_lib_src} ${tmp})
endif ()

set(shadow_lib_src ${shadow_lib_src} PARENT_SCOPE)
