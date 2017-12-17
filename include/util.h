#ifndef DISTLR_UTIL_H_
#define DISTLR_UTIL_H_

#include <string>
#include <vector>

namespace distlr {

std::vector<std::string> Split(std::string line, char sparator);

int ToInt(const char* str);

int ToInt(const std::string& str);

float ToFloat(const char* str);

float ToFloat(const std::string& str);

float CalAuc(std::vector<float>& vecPred, std::vector<int>& veclabel);

float CalLoss(std::vector<float>& vecPred, std::vector<int>& veclabel);

template <class Iter, class Compare>
void argsort(Iter iterBegin,Iter iterEnd, Compare comp, std::vector<size_t>& vecIndexs);

} // namespace distlr

#endif  // DISTLR_UTIL_H_
