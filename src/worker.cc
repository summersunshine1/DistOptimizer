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

std::vector<float> Worker::convertSparseToDense(std::vector<Feature>& vecSparse)
{
    std::vector<float> vecDense(m_nfeature, 0.0); 
    Feature feature;
    for(int i=0;i<vecSparse.size();i++)
    {
        feature = vecSparse[i];
        vecDense[feature.nfeatureid] = feature.eval;
    }
    return vecDense;
}

void Worker::train(SparseDataIter& iter, int num_iter, int batch_size)
{
    // std::cout<<"worker start train"<<std::endl;
    while (iter.HasNext()) {
        std::vector<SparseSample> batch = iter.NextBatch(batch_size);
        pullParam();
        computeSY(batch,num_iter);
        pushParam();
    }
    // std::cout<<"worker end train"<<std::endl;
}

void Worker::test(SparseDataIter& iter, int num_iter)
{
    pullParam();
    std::vector<SparseSample> batch = iter.NextBatch(-1);
    std::vector<Feature> vecFeatures;
    std::vector<float> vecPred;
    std::vector<int> veclabel;
    float tempPred = 0.0;
    float acc = 0.0;
    int label;
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& sample = batch[i];
        vecFeatures = sample.GetFeature();
        tempPred = predict(vecFeatures);
        label =  sample.GetLabel();
        if((tempPred>=0.5 && label==1) || (tempPred<0.5 && label==0))
        {
            acc+=1;
        }
        vecPred.push_back(tempPred);
        veclabel.push_back(sample.GetLabel());
    }
    float auc = distlr::CalAuc(vecPred, veclabel);
    acc = acc/batch.size();
    time_t rawtime;
    time(&rawtime);
    struct tm* curr_time = localtime(&rawtime);
    std::cout << std::setw(2) << curr_time->tm_hour << ':' << std::setw(2)
    << curr_time->tm_min << ':' << std::setw(2) << curr_time->tm_sec
    << " Iteration "<< num_iter << ", auc: " << auc << ", acc" << acc
    << std::endl;
}

float Worker::predict(std::vector<Feature>& vecfeatures)
{
    return sigmoid(vecfeatures, m_vecWeightBefore);
}

float Worker::sigmoid(std::vector<Feature>& vecFeature,std::vector<float>& vecWeight)
{
    float fres = 0;
    Feature feature;
    int nFeatureid;
    float eVal;
    for(int i=0; i<vecFeature.size(); i++)
    {
        feature = vecFeature[i];
        nFeatureid = feature.nfeatureid;
        fres += feature.eval*vecWeight[nFeatureid];
    }
    return 1.0/(1.0+exp(-fres));
}

std::vector<float> Worker::computeGradient(std::vector<SparseSample>& batch,std::vector<float>& vecWeight)
{
    std::vector<float> grad(m_vecWeightBefore.size(),0.0);
    std::vector<Feature> vecFeature;
    for(size_t i=0; i<batch.size();i++)
    {
        auto& sample = batch[i];
        vecFeature = sample.GetFeature();
        Feature feature;
        for(size_t j=0; j<vecFeature.size(); j++)
        {
            feature = vecFeature[j];
            grad[feature.nfeatureid] += (sigmoid(vecFeature,vecWeight) - sample.GetLabel()) * feature.eval;
        }
    }
    for(size_t j=0;j<m_nfeature;j++)
    {
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

void Worker::computeSY(std::vector<SparseSample>& batch,int nIter)
{
    std::vector<float> vecGradBefore = computeGradient(batch,m_vecWeightBefore);
    if(vectorAllzero(m_vecD))
    {
        std::cout<<"init...."<<ps::MyRank()<<std::endl;
        std::transform(vecGradBefore.begin(), vecGradBefore.end(), m_vecD.begin(), std::bind( std::multiplies<float>(),-1,_1));
    }
    m_vecLr.resize(m_vecD.size());
    // std::cout<<"grad beigin"<<std::endl;
    for(int i=0;i<m_vecD.size();i++)
    {
        float temp = m_Adam->getgrad(m_vecD[i], i, nIter);
        // std::cout<<temp<<" ";
        m_vecLr[i] = temp;
    }
    // std::cout<<"grad end"<<std::endl;
    // std::cout<<std::endl;
    m_vecS = m_vecLr;
    m_vecY.resize(m_vecD.size());
    m_vecWeightAfter.resize(m_vecD.size());
    // std::cout<<"updat w begin"<<std::endl;
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