if (${BUILD_SERVICE} AND Protobuf_FOUND)
  file(GLOB tmp "shadow_server.cpp")
  set(shadow_server_src ${shadow_server_src} ${tmp})

  file(GLOB tmp "shadow_client.cpp")
  set(shadow_client_src ${shadow_client_src} ${tmp})

  set(shadow_server_src ${shadow_server_src} PARENT_SCOPE)
  set(shadow_client_src ${shadow_client_src} PARENT_SCOPE)
endif ()
