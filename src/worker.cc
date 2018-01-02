#include<iostream>
#include <iomanip>
#include "cmath"
#include "worker.h"

using namespace std::placeholders;
Worker::Worker(int n_feature,float n_lamda) : m_nIteration(0)
{
    m_nfeature = n_feature;
    m_elamda = n_lamda;
    m_Adam = new Adam(m_nfeature); 
    m_l1co = 0.0001;
    m_bisL1 = false;
}

void Worker::setL1(bool isl1)
{
    m_bisL1 = isl1;
}

void Worker::firstTrain(SparseDataIter& iter,int batch_size)
{
    if(iter.HasNext()) {
        m_nIteration += 1;
        std::vector<SparseSample> batch = iter.NextBatch(batch_size);
        pullParam();
        computeSY(batch,m_nIteration);
        pushParam();
    }
}

void Worker::train(SparseDataIter& iter, int num_iter,int batch_size)
{
   
    // std::cout<<"worker start train"<<std::endl;
    if(ps::MyRank()==0)
    {
        iter.NextBatch(batch_size);
    }

    while(iter.HasNext()) {
        m_nIteration += 1;
        std::vector<SparseSample> batch = iter.NextBatch(batch_size);
        pullParam();
        computeSY(batch,m_nIteration);
        pushParam();
        if(m_nIteration%10==0)
        {
            if(ps::MyRank()==0)
            {
                std::string root = ps::Environment::Get()->find("DATA_DIR");
                std::string filename = root + "/test/part-001";
                SparseDataIter test_iter(filename);
                test(test_iter, num_iter);
            }
        } 
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
    acc = acc*1.0/batch.size();
    float loss = distlr::CalLoss(vecPred, veclabel);
    if(m_bisL1)
    {
        loss+=m_l1co * distlr::SumvecAbs(m_vecWeightBefore);
    }
    loss = loss*1.0/batch.size();
    time_t rawtime;
    time(&rawtime);
    struct tm* curr_time = localtime(&rawtime);
    std::cout << std::setw(2) << curr_time->tm_hour << ':' << std::setw(2)
    << curr_time->tm_min << ':' << std::setw(2) << curr_time->tm_sec
    << ",sample iteration " << m_nIteration<<",auc: " << auc << ", loss" << loss<<", acc"<<acc
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
    float temp = 0.0;
    for(size_t i=0; i<batch.size();i++)
    {
        auto& sample = batch[i];
        vecFeature = sample.GetFeature();
        Feature feature;
        temp = sigmoid(vecFeature,vecWeight)-sample.GetLabel();
        for(size_t j=0; j<vecFeature.size(); j++)
        {
            feature = vecFeature[j];
            grad[feature.nfeatureid] += temp * feature.eval;
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
        int minimum = 1e-8;
        if(m_bisL1)
        {
            computePesudoGrad(m_vecWeightBefore, vecGradBefore);
            std::transform(m_vecPGrad.begin(), m_vecPGrad.end(), m_vecD.begin(), std::bind( std::multiplies<float>(),-1,_1));
        }
        else{
            std::transform(vecGradBefore.begin(), vecGradBefore.end(), m_vecD.begin(), std::bind( std::multiplies<float>(),-minimum,_1));
        }
    }
    else
    {
        if(m_bisL1)
        {
            computePesudoGrad(m_vecWeightBefore, vecGradBefore);
        }
    }
    if(m_bisL1)
    {
        std::vector<float> temp(m_vecPGrad.size());
        // fix sign of direction
        std::transform(m_vecPGrad.begin(), m_vecPGrad.end(), temp.begin(), std::bind( std::multiplies<float>(),-1,_1));
        fixSign(m_vecD,temp);
    }
    m_vecLr.resize(m_vecD.size());
    for(int i=0;i<m_vecD.size();i++)
    {
        m_vecLr[i] = m_Adam->getmaxgrad(m_vecD[i], i, nIter);
    }
    //w+=lr
    m_vecWeightAfter.resize(m_vecD.size());
    std::transform (m_vecWeightBefore.begin(), m_vecWeightBefore.end(), m_vecLr.begin(), m_vecWeightAfter.begin(), std::plus<float>());
   
       
    if(m_bisL1)
    {
        std::vector<float> vecOrth = getorthat(m_vecWeightBefore);
        fixSign(m_vecWeightAfter,vecOrth);
        //s = neww-w
        m_vecS.resize(m_vecWeightAfter.size());
        std::transform(m_vecWeightAfter.begin(), m_vecWeightAfter.end(), m_vecWeightBefore.begin(), m_vecS.begin(), std::minus<float>());  
    }
    else
    {
        m_vecS = m_vecLr;
    }
    m_vecGrad = computeGradient(batch,m_vecWeightAfter);
    /*y = g'-g+lamda*s*/
    m_vecY.resize(m_vecD.size());
    /*y=g'-g*/
    std::transform(m_vecGrad.begin(), m_vecGrad.end(), vecGradBefore.begin(), m_vecY.begin(), std::minus<float>());
    
    std::vector<float> vecTemp(m_nfeature);
    /*lamda*s*/
    std::transform(m_vecS.begin(), m_vecS.end(), vecTemp.begin(), std::bind( std::multiplies<float>(),m_elamda,_1));
    /*y = y+lamda*s*/
    std::transform(m_vecY.begin(), m_vecY.end(), vecTemp.begin(), m_vecY.begin(), std::plus<float>());
}

void Worker::pushParam()
{
    std::vector<ps::Key> keys(m_nfeature*3);
    for (int i = 0; i < m_nfeature*3; ++i) {
        keys[i] = i; 
        // std::cout<<kets[i]<<" ";
    }
    // keys = std::vector<ps::Key>(keys.rbegin(),keys.rend());
    std::vector<float> vals(m_nfeature*3);
    if(!m_bisL1)
    {
        for(int i=0;i<m_nfeature;i++)
        {
            vals[i] = m_vecS[i];
            vals[i+m_nfeature] = m_vecY[i];
            vals[i+2*m_nfeature] = m_vecGrad[i];
        }
    }
    else
    {
        for(int i=0;i<m_nfeature;i++)
        {
            vals[i] = m_vecS[i];
            vals[i+m_nfeature] = m_vecY[i];
            vals[i+2*m_nfeature] = m_vecPGrad[i];
        }
    }
    // vals = std::vector<float>(vals.rbegin(),vals.rend());
    // std::cout<<std::endl<<"push..."<<vals.size()<<std::endl;
    // std::cout<<vals[0]<<" "<<vals[m_nfeature*3-1]<<std::endl;
    
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

void Worker::computePesudoGrad(std::vector<float>& vecWeight,std::vector<float>& vecGrad)
{
    int l = vecWeight.size();
    m_vecPGrad.resize(0);
    m_vecPGrad.resize(l,0.0);
    for(int i=0;i<l;i++)
    {
        if(vecWeight[i]<0.0)
        {
            m_vecPGrad[i]=vecGrad[i]-m_l1co;
        }
        else if(vecWeight[i]>0.0)
        {
            m_vecPGrad[i]=vecGrad[i]+m_l1co;
        }
        else
        {
            if(vecGrad[i]+m_l1co<0)
            {
                m_vecPGrad[i] = vecGrad[i]+m_l1co;
            }
            else if(vecGrad[i]-m_l1co>0)
            {
                m_vecPGrad[i] = vecGrad[i]-m_l1co;
            }
        }
    }
}

void Worker::fixSign(std::vector<float>& vec,std::vector<float>& vecOrth)
{
    for(int i=0;i<vec.size();i++)
    {
        if(vec[i]*vecOrth[i]<0)
        {
            vec[i]=0.0;
        }
    }
}

std::vector<float> Worker::getorthat(std::vector<float>& vecWeight)
{
    std::vector<float> vecOrth(vecWeight.size());
    for(int i=0;i<vecWeight.size();i++)
    {
        if(vecWeight[i]!=0)
        {
            if(vecWeight[i]<0)
            {
                vecOrth[i] = -1.0;
            }
            else{
                vecOrth[i] = 1.0;
            }
        }
        else
        {
            if(m_vecPGrad[i]<0)
            {
                vecOrth[i]=1.0;
            }
            else if(m_vecPGrad[i]>0)
            {
                vecOrth[i]=-1.0;
            }
        }
    }
    return vecOrth;
}

