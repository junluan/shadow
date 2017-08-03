file(GLOB_RECURSE tmp *.cpp *.hpp)
set(shadow_tools_src ${shadow_tools_src} ${tmp})

file(GLOB tmp "../util/io.cpp")
set(shadow_tools_src ${shadow_tools_src} ${tmp})

set(shadow_tools_src ${shadow_tools_src} PARENT_SCOPE)
