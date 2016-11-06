add_executable(caffe2shadow "caffe2shadow.cpp" "../src/shadow/util/io.cpp")
target_link_libraries(caffe2shadow glog proto)
