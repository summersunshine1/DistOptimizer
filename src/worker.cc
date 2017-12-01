#include<iostream>
#include <iomanip>
#include "cmath"
#include "worker.h"

using namespace std::placeholders;
Worker::Worker(int n_feature,float n_lamda)
{
    m_nfeature = n_feature;
    m_elamda = n_lamda;
    m_Adam = new Adam(m_nfeature); 
}

void Worker::train(DataIter& iter, int num_iter, int batch_size)
{
    while (iter.HasNext()) {
        std::vector<Sample> batch = iter.NextBatch(batch_size);
        pullParam();
        computeSY(batch,num_iter);
        pushParam();
    }
}

void Worker::test(DataIter& iter, int num_iter)
{
    pullParam();
    std::vector<Sample> batch = iter.NextBatch(-1);
    float acc = 0;
    std::vector<float> vecFeatures;
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& sample = batch[i];
        vecFeatures = sample.GetFeature();
        if (predict(vecFeatures) == sample.GetLabel()) {
            ++acc;
        }
    }
    time_t rawtime;
    time(&rawtime);
    struct tm* curr_time = localtime(&rawtime);
    std::cout << std::setw(2) << curr_time->tm_hour << ':' << std::setw(2)
    << curr_time->tm_min << ':' << std::setw(2) << curr_time->tm_sec
    << " Iteration "<< num_iter << ", accuracy: " << acc / batch.size()
    << std::endl;
}

int Worker::predict(std::vector<float>& vecfeatures)
{
    float z = 0;
    for (size_t j = 0; j < m_vecWeightBefore.size(); ++j) {
        // std::cout<<m_vecWeightBefore[j]<<" ";
        z += m_vecWeightBefore[j] * vecfeatures[j];
    }
    // std::cout<<std::endl;
    return z > 0;
}

float Worker::sigmoid(std::vector<float>& vecFeature,std::vector<float>& vecWeight)
{
    float fres = 0;
    for (int i = 0;i<m_nfeature;i++)
    {
        fres += vecFeature[i]*vecWeight[i];
    }
    return 1.0/(1.0+exp(-fres));
}

std::vector<float> Worker::computeGradient(std::vector<Sample>& batch,std::vector<float>& vecWeight)
{
    std::vector<float> grad(m_vecWeightBefore.size());
    std::vector<float> vecFeature;
    for (size_t j = 0; j < m_nfeature; ++j) {
      grad[j] = 0;
      for (size_t i = 0; i < batch.size(); ++i) {
        auto& sample = batch[i];
        vecFeature = sample.GetFeature();
        grad[j] += (sigmoid(vecFeature,vecWeight) - sample.GetLabel()) * sample.GetFeature(j);
      }
      grad[j] = 1. * grad[j] / batch.size();
    }
    return grad; 
}

bool Worker::vectorAllzero(std::vector<float>& vec)
{
    bool bZero = true;
    for(int i =0;i<vec.size();i++)
    {
        if(vec[i]!=0)
        {
            bZero = false;
            return bZero;
        }
    }
    return bZero;
}

void Worker::computeSY(std::vector<Sample>& batch,int nIter)
{
    std::vector<float> vecGradBefore = computeGradient(batch,m_vecWeightBefore);
    if(vectorAllzero(m_vecD))
    {
        std::cout<<"init...."<<ps::MyRank()<<std::endl;
        std::transform(vecGradBefore.begin(), vecGradBefore.end(), m_vecD.begin(), std::bind( std::multiplies<float>(),-1,_1));
    }
    m_vecLr.resize(m_vecD.size());
    for(int i=0;i<m_vecD.size();i++)
    {
        float temp = m_Adam->getgrad(m_vecD[i], i, nIter);
        // std::cout<<temp<<" ";
        m_vecLr[i] = temp;
    }
    // std::cout<<std::endl;
    m_vecS = m_vecLr;
    m_vecY.resize(m_vecD.size());
    m_vecWeightAfter.resize(m_vecD.size());
    
    // std::transform(m_vecLr.begin(), m_vecLr.end(), m_vecS.begin(), std::bind( std::multiplies<float>(),-1,_1));
    std::transform (m_vecWeightBefore.begin(), m_vecWeightBefore.end(), m_vecS.begin(), m_vecWeightAfter.begin(), std::plus<float>());
    m_vecGrad = computeGradient(batch,m_vecWeightAfter);
    /*y = g'-g+lamda*s*/
    std::transform(m_vecGrad.begin(), m_vecGrad.end(), vecGradBefore.begin(), m_vecY.begin(), std::minus<float>());
    
    std::vector<float> vecTemp(m_nfeature);
    std::transform(m_vecS.begin(), m_vecS.end(), vecTemp.begin(), std::bind( std::multiplies<float>(),m_elamda,_1));
    std::transform(m_vecY.begin(), m_vecY.end(), vecTemp.begin(), m_vecY.begin(), std::plus<float>());
}

void Worker::pushParam()
{
    std::vector<ps::Key> keys(m_nfeature*3);
    for (int i = 0; i < m_nfeature*3; ++i) {
        keys[i] = i;
    }
    std::vector<float> vals(m_nfeature*3);
    for(int i=0;i<m_nfeature;i++)
    {
        vals[i] = m_vecS[i];
        vals[i+m_nfeature] = m_vecY[i];
        vals[i+2*m_nfeature] = m_vecGrad[i];
    }
    m_kv->Wait(m_kv->Push(keys, vals));
}

void Worker::pullParam()
{
    std::vector<ps::Key> keys(m_nfeature*2);
    std::vector<float> vals;
    for (int i = 0; i < m_nfeature*2; ++i) {
        keys[i] = i;
    }
    m_kv->Wait(m_kv->Pull(keys, &vals));
    m_vecWeightBefore = std::vector<float>(vals.begin(),vals.end()-m_nfeature);
    
    m_vecD = std::vector<float>(vals.begin()+m_nfeature,vals.end());
}