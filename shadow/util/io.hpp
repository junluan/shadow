#ifndef SHADOW_CORE_UTIL_IO_HPP
#define SHADOW_CORE_UTIL_IO_HPP

#if defined(USE_Protobuf)
#include <google/protobuf/message.h>

#if GOOGLE_PROTOBUF_VERSION >= 3001000
#define SUPPORT_JSON
#endif

#else
#include "core/params.hpp"
#endif

namespace Shadow {

#if defined(USE_Protobuf)
namespace IO {

using google::protobuf::Message;

bool ReadProtoFromText(const std::string& proto_text, Message* proto);

bool ReadProtoFromTextFile(const std::string& proto_file, Message* proto);

bool ReadProtoFromBinaryFile(const std::string& proto_file, Message* proto);

void WriteProtoToText(const Message& proto, std::string* proto_text);

void WriteProtoToTextFile(const Message& proto, const std::string& proto_file);

void WriteProtoToBinaryFile(const Message& proto,
                            const std::string& proto_file);

#if defined(SUPPORT_JSON)
void WriteProtoToJsonText(const Message& proto, std::string* json_text,
                          bool compact = false);
#endif

}  // namespace IO

#else
namespace IO {

bool ReadProtoFromText(const std::string& proto_text,
                       shadow::NetParameter* proto);

bool ReadProtoFromTextFile(const std::string& proto_file,
                           shadow::NetParameter* proto);

}  // namespace IO
#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_UTIL_IO_HPP
