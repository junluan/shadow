file(GLOB_RECURSE tmp "*.cpp" "*.hpp")
set(shadow_lib_src ${shadow_lib_src} ${tmp})

if (${USE_CUDA})
  file(GLOB_RECURSE tmp "*.cu")
  set(shadow_lib_src ${shadow_lib_src} ${tmp})
endif ()

set(shadow_lib_src ${shadow_lib_src} PARENT_SCOPE)
