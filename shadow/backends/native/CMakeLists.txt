file(GLOB_RECURSE tmp "*.cpp" "*.hpp")
set(shadow_lib_src ${shadow_lib_src} ${tmp})

set(shadow_lib_src ${shadow_lib_src} PARENT_SCOPE)
