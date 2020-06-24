#ifndef SHADOW_UTIL_IO_HPP_
#define SHADOW_UTIL_IO_HPP_

#if defined(USE_Protobuf)
#include <google/protobuf/message.h>

#if GOOGLE_PROTOBUF_VERSION >= 3003000
#define SUPPORT_JSON
#endif

#else
#include "core/params.hpp"
#endif

namespace Shadow {

namespace IO {

#if defined(USE_Protobuf)
using google::protobuf::Message;

bool ReadProtoFromText(const std::string& proto_text, Message* proto);

bool ReadProtoFromTextFile(const std::string& proto_file, Message* proto);

bool ReadProtoFromBinaryFile(const std::string& proto_file, Message* proto);

bool ReadProtoFromArray(const void* proto_data, int proto_size, Message* proto);

bool WriteProtoToText(const Message& proto, std::string* proto_text);

bool WriteProtoToTextFile(const Message& proto, const std::string& proto_file);

bool WriteProtoToBinaryFile(const Message& proto,
                            const std::string& proto_file);

bool WriteProtoToArray(const Message& proto, void* proto_data);

#if defined(SUPPORT_JSON)
bool WriteProtoToJsonText(const Message& proto, std::string* json_text,
                          bool compact = false);
#endif

#else
bool ReadProtoFromText(const std::string& proto_text, shadow::NetParam* proto);

bool ReadProtoFromTextFile(const std::string& proto_file,
                           shadow::NetParam* proto);
#endif

}  // namespace IO

}  // namespace Shadow

#endif  // SHADOW_UTIL_IO_HPP_
