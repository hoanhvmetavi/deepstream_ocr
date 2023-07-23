#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <fstream>
std::vector<std::string> dict_table;
bool dict_ready = false;

bool read_dict()
{
  std::ifstream fdict;
  setlocale(LC_CTYPE, "");
  fdict.open("/opt/nvidia/deepstream/deepstream-6.1/surveillance_ai/models/ocr/jp_dict.txt");
  if (!fdict.is_open())
  {
    std::cout << "open dictionary file failed." << std::endl;
    return false;
  }
  while (!fdict.eof())
  {
    std::string strLineAnsi;
    if (getline(fdict, strLineAnsi))
    {
      if (strLineAnsi.length() > 1)
      {
        strLineAnsi.erase(1);
      }
      dict_table.push_back(strLineAnsi);
    }
  }

  fdict.close();
  return true;
}

extern "C" bool NvDsInferParseCustomOCR(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString)

{

  if (!dict_ready)
  {
    if (read_dict())
      std::cout << "load success" << std::endl;
    dict_ready = true;
  }

  auto layerFinder =
      [&outputLayersInfo](
          const std::string &name) -> const NvDsInferLayerInfo *
  {
    for (auto &layer : outputLayersInfo)
    {

      if ((layer.layerName && name == layer.layerName))
      {
        return &layer;
      }
    }
    return nullptr;
  };
  const NvDsInferLayerInfo *output_dets = layerFinder("OUTPUT_DETS");
  int h = output_dets->inferDims.d[1]; // #heatmap.size[2];
  int w = output_dets->inferDims.d[2]; // heatmap.size[3];
  int c = output_dets->inferDims.d[0];
  int spacial_size = w * h * c;
  float *output_dets_arr = (float *)(output_dets->buffer);
  const NvDsInferLayerInfo *output_text_recs = layerFinder("TEXTS_RECS");
  int num_texts = output_text_recs->inferDims.d[0];
  int len_text = output_text_recs->inferDims.d[1];
  int *text_rects = (int *)output_text_recs->buffer;
  const NvDsInferLayerInfo *output_text_scores = layerFinder("SCORES_RECS");
  float *text_scores = (float *)output_text_scores->buffer;
  for (int k = 0; k < 1; k++)
  {
    attrString = "";
    for (int l = 0; l < 20; l++)
    {
      if (text_rects[k * 20 + l] == -1)
        break;
      else
      {
        attrString += dict_table[text_rects[k * 20 + l]];
      }
    }

    // std::cout << "text:" << attrString << " text_rects[k * 20 + l]: " << text_rects[k * 20 + l] << std::endl;
    NvDsInferAttribute att;
    att.attributeIndex = 0;
    att.attributeConfidence = (float)text_scores[k];
    att.attributeLabel = strdup(attrString.c_str());
    attrList.push_back(att);
  }

  return true;
}
