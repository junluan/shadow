file(GLOB yolo_src "yolo/*.cpp")
add_executable(yolo ${yolo_src})
target_link_libraries(yolo shadow)

file(GLOB ssd_src "ssd/*.cpp")
add_executable(ssd ${ssd_src})
target_link_libraries(ssd shadow)
