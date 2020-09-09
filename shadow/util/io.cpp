#include "io.hpp"

#include "log.hpp"

#if defined(USE_Protobuf)
#include <fstream>
#include <memory>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#if defined(SUPPORT_JSON)
#include <google/protobuf/util/json_util.h>
#endif

#else
#include "core/parser.hpp"
#include "util/util.hpp"
#endif

namespace Shadow {

namespace IO {

#if defined(USE_Protobuf)
using google::protobuf::TextFormat;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::IstreamInputStream;
using google::protobuf::io::OstreamOutputStream;

bool ReadProtoFromText(const std::string& proto_text, Message* proto) {
  return TextFormat::ParseFromString(proto_text, proto);
}

bool ReadProtoFromTextFile(const std::string& proto_file, Message* proto) {
  std::ifstream ifs(proto_file);
  CHECK(ifs.is_open()) << "File not found: " << proto_file;
  auto istream_input = std::make_shared<IstreamInputStream>(&ifs);
  return TextFormat::Parse(istream_input.get(), proto);
}

bool ReadProtoFromBinaryFile(const std::string& proto_file, Message* proto) {
  std::ifstream ifs(proto_file, std::ios::binary);
  CHECK(ifs.is_open()) << "File not found: " << proto_file;
  auto istream_input = std::make_shared<IstreamInputStream>(&ifs);
  auto coded_input = std::make_shared<CodedInputStream>(istream_input.get());
  coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
  return proto->ParseFromCodedStream(coded_input.get()) &&
         coded_input->ConsumedEntireMessage();
}

bool ReadProtoFromArray(const void* proto_data, int proto_size,
                        Message* proto) {
  auto coded_input = std::make_shared<CodedInputStream>(
      static_cast<const uint8_t*>(proto_data), proto_size);
  coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
  return proto->ParseFromCodedStream(coded_input.get()) &&
         coded_input->ConsumedEntireMessage();
}

bool WriteProtoToText(const Message& proto, std::string* proto_text) {
  return TextFormat::PrintToString(proto, proto_text);
}

bool WriteProtoToTextFile(const Message& proto, const std::string& proto_file) {
  std::ofstream ofs(proto_file);
  CHECK(ofs.is_open()) << "File open failed: " << proto_file;
  auto ostream_output = std::make_shared<OstreamOutputStream>(&ofs);
  return TextFormat::Print(proto, ostream_output.get());
}

bool WriteProtoToBinaryFile(const Message& proto,
                            const std::string& proto_file) {
  std::ofstream ofs(proto_file, std::ios::binary);
  CHECK(ofs.is_open()) << "File open failed: " << proto_file;
  auto ostream_output = std::make_shared<OstreamOutputStream>(&ofs);
  return proto.SerializeToZeroCopyStream(ostream_output.get());
}

bool WriteProtoToArray(const Message& proto, void* proto_data) {
  return proto.SerializeToArray(proto_data, proto.ByteSize());
}

#if defined(SUPPORT_JSON)
using google::protobuf::util::JsonPrintOptions;
using google::protobuf::util::MessageToJsonString;

bool WriteProtoToJsonText(const Message& proto, std::string* json_text,
                          bool compact) {
  JsonPrintOptions options;
  options.add_whitespace = !compact;
  options.preserve_proto_field_names = true;
  return MessageToJsonString(proto, json_text, options).ok();
}
#endif

#else
bool ReadProtoFromText(const std::string& proto_text, shadow::NetParam* proto) {
  Parser::ParseNet(proto_text, proto);
  return true;
}

bool ReadProtoFromTextFile(const std::string& proto_file,
                           shadow::NetParam* proto) {
  return ReadProtoFromText(Util::read_text_from_file(proto_file), proto);
}
#endif

}  // namespace IO

}  // namespace Shadow
