#ifndef SHADOW_UTIL_PARSER_HPP
#define SHADOW_UTIL_PARSER_HPP

#include "shadow/network.hpp"

#include <string>

class Parser {
 public:
  void ParseNetworkProtoTxt(Network *net, const std::string proto_txt,
                            int batch);
  void ParseNetworkProtoStr(Network *net, const std::string proto_str,
                            int batch);
  void LoadWeights(Network *net, const std::string weight_file);
  void LoadWeights(Network *net, const float *weights);

 private:
  void ParseNet(Network *net);
  Layer *LayerFactory(const shadow::LayerParameter &layer_param,
                      VecBlob *blobs);

  void LoadWeightsUpto(Network *net, const std::string weight_file,
                       int cut_off);
  void LoadWeightsUpto(Network *net, const float *weights, int cut_off);
};

#endif  // SHADOW_UTIL_PARSER_HPP
