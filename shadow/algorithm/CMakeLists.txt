file(GLOB_RECURSE tmp "*.cpp" "*.hpp")
set(shadow_algorithm_src ${shadow_algorithm_src} ${tmp})

set(shadow_algorithm_src ${shadow_algorithm_src} PARENT_SCOPE)
