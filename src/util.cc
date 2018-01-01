#include "util.h"
#include <iostream>
#include <algorithm>
#include <math.h>

namespace distlr {

std::vector<std::string> Split(std::string line, char sparator) {
  std::vector<std::string> ret;
  int start = 0;
  std::size_t pos = line.find(sparator, start);
  while (pos != std::string::npos) {
    ret.push_back(line.substr(start, pos-start));
    start = pos + 1;
    pos = line.find(sparator, start);
  }
  ret.push_back(line.substr(start));
  return ret;
}

int ToInt(const char* str) {
  int flag = 1, ret = 0;
  const char* p = str;

  if (*p == '-') {
    ++p;
    flag = -1;
  } else if (*p == '+') {
    ++p;
  }

  while (*p) {
    ret = ret * 10 + (*p - '0');
    ++p;
  }
  return flag * ret;
}

int ToInt(const std::string& str) {
  return ToInt(str.c_str());
}

float ToFloat(const char* str) {
  float integer = 0, decimal = 0;
  float base = 1;
  const char* p = str;

  while (*p) {
    if (*p == '.') {
      base = 0.1;
      ++p;
      continue;
    }
    if (base >= 1.0) {
      integer = integer * 10 + (*p - '0');
    } else {
      decimal += base * (*p - '0');
      base *= 0.1;
    }
    ++p;
  }

  return integer + decimal;
}

float ToFloat(const std::string& str) {
  return ToFloat(str.c_str());
}
template <class Iter, class Compare>
void argsort(Iter iterBegin,Iter iterEnd, Compare comp, std::vector<size_t>& vecIndexs)
{
    std::vector<std::pair<size_t, Iter>> pv;
    Iter iter;
    size_t k;
    for(iter = iterBegin,k=0;iter != iterEnd; iter++,k++)
    {
        pv.push_back(std::pair<int, Iter>(k, iter));
    }
    std::sort(pv.begin(),pv.end(), [&comp](const std::pair<size_t, Iter>&a,const std::pair<size_t,Iter>&b)->bool
    {return comp(*a.second, *b.second);});
    vecIndexs.resize(pv.size());
    std::transform(pv.begin(), pv.end(), vecIndexs.begin(), 
        [](const std::pair<size_t,Iter>& a) -> size_t { return a.first ; });
}

float CalAuc(std::vector<float>& vecPred, std::vector<int>& veclabel)
{
    std::vector<size_t> vecIndexs;
    argsort(vecPred.begin(), vecPred.end(), std::greater<float>(), vecIndexs);
    int tp = 0;
    int fp = 0;
    int lastfp = 0;
    int lasttp = 0;
    int index;
    float auc = 0.0;
    float lastscore = vecPred[vecIndexs[0]]+1;
    for(int i=0;i<vecIndexs.size();i++)
    {
        index = vecIndexs[i];
        if(lastscore != vecPred[index])
        {
            auc += (tp+lasttp)*(fp-lastfp)*1.0/2;
            lastscore = vecPred[index];
            lasttp = tp;
            lastfp = fp;
        }
        // std::cout<<vecPred[index]<<" ";
        if(veclabel[index]==1)
        {   
            tp += 1;
        }
        else
        {
            fp += 1;
        }
    }

    if(tp==0 || tp==vecIndexs.size())
    {
        return 0.0;
    }
    auc += (tp+lasttp)*(fp-lastfp)*1.0/2;
    auc = auc*1.0/(tp*fp);
    std::cout<<std::endl;
    return auc;
}

float CalLoss(std::vector<float>& vecPred, std::vector<int>& veclabel)
{
    float loss = 0.0;
    for(int i=0;i<vecPred.size();i++)
    {
        loss+=veclabel[i]*log(vecPred[i])+(1-veclabel[i])*log(1-vecPred[i]);
    }
    return -1.0*loss;
}

float SumvecAbs(std::vector<float>& w)
{
    float temp =0.0;
    for(int i=0;i<w.size();i++)
    {
        if(w[i]<0)
        {
            temp+=-1*w[i];
        }
        else
        {
            temp+=w[i];
        }
    }
    return temp;
}

template <class T>
void clearVector(std::vector<T>& v)
{
    std::vector<T>().swap(v);
}

} // namespace distlr
