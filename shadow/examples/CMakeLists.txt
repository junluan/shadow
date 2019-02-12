file(GLOB tmp "demo_*.cpp" "classify.cpp" "detect_*.cpp" "*.hpp")
set(shadow_examples_src ${shadow_examples_src} ${tmp})

set(shadow_examples_src ${shadow_examples_src} PARENT_SCOPE)
