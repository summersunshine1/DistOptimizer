#ifndef SPARSE_SAMPLE_H_
#define SPARSE_SAMPLE_H_

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

struct Feature
{
    int nfeild;
    int nfeatureid;
    float eval;
};

class SparseSample {
public:
  explicit SparseSample(){
  }
  explicit SparseSample(std::vector<Feature>& feature, int label)
    : m_vecFeaturs(feature), m_nlabel(label) {
  }
  virtual ~SparseSample() {
  }

  void SetLabel(int label) {
    m_nlabel = label;
  }

  void SetFeatures(const std::vector<Feature>& feature) {
    m_vecFeaturs = feature;
  }

  std::pair<std::vector<Feature>, int> GetSample() {
    return std::make_pair(m_vecFeaturs, m_nlabel);
  }

  std::vector<Feature> GetFeature() {
    return m_vecFeaturs;
  }

  int GetLabel() {
    return m_nlabel;
  }

private:
  int m_nlabel;
  std::vector<Feature> m_vecFeaturs;
};


#endif // SPARSE_SAMPLE_H_
