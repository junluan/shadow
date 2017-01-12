#ifndef SHADOW_UTIL_IO_HPP
#define SHADOW_UTIL_IO_HPP

#include <google/protobuf/message.h>

#if GOOGLE_PROTOBUF_VERSION >= 3000000
#define SUPPORT_JSON
#endif

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

#endif  // SHADOW_UTIL_IO_HPP
