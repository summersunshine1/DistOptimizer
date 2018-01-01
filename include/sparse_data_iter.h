#ifndef SPARSE_DATA_ITER_H_
#define SPARSE_DATA_ITER_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "sparse_sample.h"
#include "util.h"

class SparseDataIter {
public:
  explicit SparseDataIter(std::string filename)
    : filename_(filename),offset_(0),
      round_end_(false) { 
      std::ifstream input(filename_.c_str());
      std::string line, buf;
      samples_.clear();
      if(!input.is_open())
         return;
      while (std::getline(input, line)) {
        std::istringstream in(line);
        in >> buf;
        int label = distlr::ToInt(buf) == 1 ? 1 : 0;
        std::vector<Feature> vecfeature;
        while (in >> buf) {
          auto ss = distlr::Split(buf, ':');
          Feature feature;
          if(ss.size()==3)
          {
              feature.nfeild = distlr::ToInt(ss[0]);
              feature.nfeatureid = distlr::ToInt(ss[1])-1;
              feature.eval = distlr::ToFloat(ss[2]);
          }
          else{
              feature.nfeatureid = distlr::ToInt(ss[0]);
              feature.eval = distlr::ToFloat(ss[1]);
          }
          vecfeature.push_back(feature);
        }
        samples_.push_back(SparseSample(vecfeature, label));
      }
      // std::random_shuffle ( samples_.begin(), samples_.end() );
  }
  virtual ~SparseDataIter() {
  }

  // batch_size = -1 means all samples
  std::vector<SparseSample> NextBatch(int batch_size=100) {
    if (batch_size < 0) {
      batch_size = (int)samples_.size();
    }

    std::vector<SparseSample> batch;
    for (int i = 0; i < batch_size; ++i) {
      batch.push_back(samples_[offset_]);
      ++offset_;
      if (offset_ == (int)samples_.size()) {
        offset_ = 0;
        round_end_ = true;
      }
    }
    return batch;
  }

  bool HasNext() {
    return !round_end_;
  }

private:
  std::string filename_;
  int offset_;
  bool round_end_;
  std::vector<SparseSample> samples_;
};


#endif  // SPARSE_DATA_ITER_H_
