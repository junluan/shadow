file(GLOB tmp demo_*.cpp detection_*.cpp classification.cpp)
set(shadow_examples_src ${shadow_examples_src} ${tmp})

set(shadow_examples_src ${shadow_examples_src} PARENT_SCOPE)
