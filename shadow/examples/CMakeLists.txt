file(GLOB_RECURSE tmp *.cpp)
set(shadow_examples_src ${shadow_examples_src} ${tmp})

set(shadow_examples_src ${shadow_examples_src} PARENT_SCOPE)
