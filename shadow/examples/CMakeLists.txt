file(GLOB_RECURSE tmp *.cpp)
set(shadow_example_src ${shadow_example_src} ${tmp})

set(shadow_example_src ${shadow_example_src} PARENT_SCOPE)
