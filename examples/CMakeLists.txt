file(GLOB ssd_src "ssd/ssd.cpp")
file(GLOB demo_src "main.cpp" "demo.cpp")

add_executable(demo ${demo_src} ${ssd_src})
target_link_libraries(demo shadow)
