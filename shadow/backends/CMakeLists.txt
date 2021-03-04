file(GLOB backends_dir "*")

foreach (backend_dir ${backends_dir})
  if (IS_DIRECTORY ${backend_dir} AND EXISTS ${backend_dir}/CMakeLists.txt)
    add_subdirectory(${backend_dir})
  endif ()
endforeach ()

set(shadow_lib_src ${shadow_lib_src} PARENT_SCOPE)
