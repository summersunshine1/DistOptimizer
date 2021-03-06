#include <iostream>
#include <algorithm>
#include <functional>

#include "distserver.h"
#include "util.h"

using namespace std::placeholders;
DistServer::DistServer(float lamda):m_eEpisilo(1e-8),m_eScale(1)
{
    using namespace std::placeholders;
    init();
    m_eLamda = lamda;
    m_psServer = new ps::KVServer<float>(0);
    m_psServer->set_request_handle(
      std::bind(&DistServer::dataHandle, this, _1, _2, _3));
    m_bSync = !strcmp(ps::Environment::Get()->find("SYNC_MODE"), "1");
    std::string mode = m_bSync ? "sync" : "async";
    std::cout << "Server mode: " << mode << std::endl;
}

void DistServer::init()
{
    m_nFeature = distlr::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));
    m_vecWeight.resize(m_nFeature);
    for(int i=0;i<m_nFeature;i++)
    {
        m_vecWeight[i]=0.0;
    }
    // srand(12);
      // for (size_t i = 0; i < m_vecWeight.size(); ++i) {
        // m_vecWeight[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      // }
    m_vecGrad.resize(m_nFeature);
    m_nWindow = distlr::ToInt(ps::Environment::Get()->find("WINDOW_SIZE"));
}

void DistServer::dataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<float>& req_data,
                  ps::KVServer<float>* server) {

    size_t n = req_data.keys.size();
    if (req_meta.push) {
        // std::cout<<"handle push begin...."<<std::endl;
        CHECK_EQ(m_nFeature*3, n);
        if(m_vecDirection.size()==0)
        {
            std::vector<float> vecTemps;
            std::vector<float> vecTempy;
            /* vecs,vecy,vecgrad*/
            for(size_t i = 0; i < m_nFeature; ++i)
            {
                vecTemps.push_back(req_data.vals[i]*1.0/m_eScale);
                vecTempy.push_back(req_data.vals[i+m_nFeature]);
                m_vecGrad[i] = req_data.vals[i+m_nFeature*2];
                m_vecWeight[i] += vecTemps[i];
            }
            initFirstDirection();
            pushSY(vecTemps, vecTempy);
            updateSY();
            server->Response(req_meta);
        }
        else if(!m_bSync)
        {
            std::vector<float> vecTemps;
            std::vector<float> vecTempy;
            /* vecs,vecy,vecgrad*/
            for(size_t i = 0; i < m_nFeature; ++i)
            {
                vecTemps.push_back(req_data.vals[i]*1.0/m_eScale);
                vecTempy.push_back(req_data.vals[i+m_nFeature]);
                m_vecGrad[i] = req_data.vals[i+m_nFeature*2];
                m_vecWeight[i] += vecTemps[i];
            }
            pushSY(vecTemps, vecTempy);
            updateSY();
            server->Response(req_meta);
        }
        else
        {
            if (mergebuf.vals.empty()) {
                mergebuf.vals.resize(m_nFeature*3, 0);
            }
            for(size_t i = 0; i < m_nFeature; ++i)
            {
                mergebuf.vals[i]+=req_data.vals[i];
                mergebuf.vals[i+m_nFeature]+=req_data.vals[i+m_nFeature];
                mergebuf.vals[i+m_nFeature*2]+=req_data.vals[i+m_nFeature*2];
            }
            mergebuf.request.push_back(req_meta);
            int workers = (size_t)ps::NumWorkers();
            if(mergebuf.request.size()==workers)
            {
                std::vector<float> vecTemps;
                std::vector<float> vecTempy;
                for(size_t i = 0; i < m_nFeature; ++i)
                {
                    vecTemps.push_back(mergebuf.vals[i]*1.0/workers);
                    vecTempy.push_back(mergebuf.vals[i+m_nFeature]*1.0/workers);
                    m_vecGrad[i] = mergebuf.vals[i+m_nFeature*2]*1.0/workers;
                    // std::cout<<vecTemps[i]<<" ";
                    m_vecWeight[i] += vecTemps[i];
                }
                pushSY(vecTemps, vecTempy);
                updateSY();
                for(size_t i=0;i<workers;i++)
                {
                    server->Response(mergebuf.request[i]);
                }
                mergebuf.request.clear();
                mergebuf.vals.clear();
                // distlr::clearVector(mergebuf.request);
                // distlr::clearVector(mergebuf.vals);
            }
        }
        // std::cout<<"handle push end...."<<std::endl;
     }
    else { // pull
        // std::cout<<"pull....."<<m_vecDirection.size()<<std::endl;
        ps::KVPairs<float> response;
        response.keys = req_data.keys;
        response.vals.resize(n);
        if(m_vecDirection.size()!=0)
        {            
            for (size_t i = 0; i < m_nFeature; ++i) {
                response.vals[i] = m_vecWeight[i];
                response.vals[i+m_nFeature] = m_vecDirection[i];
            }
        }
        else
        {
            for (size_t i = 0; i < m_nFeature; ++i) {
                response.vals[i] = m_vecWeight[i];
                response.vals[i+m_nFeature] = 0;   
            }
        }
        server->Response(req_meta, response);
    }
}
    
int DistServer::DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
}

void DistServer::pop_front(std::vector<std::vector<float>>& vec)
{
    assert(!vec.empty());
    vec.erase(vec.begin());
}

void DistServer::pushSY(std::vector<float>& vecTemps, std::vector<float>& vecTempy)
{
    if(m_vecS.size()>=m_nWindow)
    {
        pop_front(m_vecS);
        pop_front(m_vecY);
    }
    m_vecS.push_back(vecTemps);
    m_vecY.push_back(vecTempy);
}

void DistServer::updateSY(){
    int nSize = m_vecS.size();
    int temp = nSize-1;
    std::vector<float> vecAlpha;
    float eTempAlpha = 0.0;
    std::vector<float> vecGrad = m_vecGrad;
    std::vector<float> vecMiddle(m_nFeature);
    while(temp>=0)
    {
        float num = std::inner_product(m_vecS[temp].begin(), m_vecS[temp].end(), vecGrad.begin(), 0.0);//s*q
        float dom = std::inner_product(m_vecY[temp].begin(), m_vecY[temp].end(), m_vecS[temp].begin(), 0.0);//y*s
        eTempAlpha = num/(dom+m_eEpisilo);//a=s*q/(y*s+e)
        vecAlpha.push_back(eTempAlpha);
        transform(m_vecY[temp].begin(), m_vecY[temp].end(), vecMiddle.begin(),std::bind( std::multiplies<float>(),eTempAlpha,_1));//y*alpha
        transform(vecGrad.begin(), vecGrad.end(),vecMiddle.begin(),vecGrad.begin(), std::minus<float>());//q=q-y*alpha
        temp-=1;
    }
    temp = nSize-1;
    float fnum = std::inner_product(m_vecS[temp].begin(), m_vecS[temp].end(), m_vecY[temp].begin(), 0.0);
    float fdom = std::inner_product(m_vecY[temp].begin(), m_vecY[temp].end(), m_vecY[temp].begin(), 0.0);
    transform(vecGrad.begin(), vecGrad.end(), vecGrad.begin(),std::bind( std::multiplies<float>(),fnum/(fdom+m_eEpisilo),_1));
    float beta = 0.0;
    for(temp = 0;temp<nSize;temp++)
    {
        float num = std::inner_product(m_vecY[temp].begin(), m_vecY[temp].end(), vecGrad.begin(), 0.0);//y*q
        float dom = std::inner_product(m_vecY[temp].begin(), m_vecY[temp].end(), m_vecS[temp].begin(), 0.0);//y*s
        beta = (num)/(dom+m_eEpisilo);//beta=y*q/(y*s+e)
        float ftemp = m_eScale * vecAlpha[nSize-1-temp]-beta;//c*alpha-beta
        //s*(c*alpha-beta)
        transform(m_vecS[temp].begin(), m_vecS[temp].end(), vecMiddle.begin(), std::bind( std::multiplies<float>(),ftemp,_1));
        //q = q+s*(c*alpha-beta)
        transform(vecGrad.begin(), vecGrad.end(),vecMiddle.begin(),vecGrad.begin(), std::plus<float>());
    }
    //yk*sk
    float ftemp = std::inner_product(m_vecY[nSize-1].begin(), m_vecY[nSize-1].end(), m_vecS[nSize-1].begin(), 0.0);
    if(ftemp>0)
    {
        //d = -q
        transform(vecGrad.begin(), vecGrad.end(),m_vecDirection.begin(), std::bind( std::multiplies<float>(),-1,_1));
    }
    
}

void DistServer::initFirstDirection()
{
    //
    m_vecDirection.resize(m_nFeature);
    transform(m_vecGrad.begin(), m_vecGrad.end(), m_vecDirection.begin(), std::bind( std::multiplies<float>(),-1,_1));
    // std::vector<float> vecTemp(m_nFeature);
    // transform(vecs.begin(), vecs.end(), vecTemp.begin(), std::bind( std::multiplies<float>(),m_eLamda,_1));
    // transform(vecGrad.begin(), vecGrad.end(), vecTemp.begin(),vecTemp.begin(), std::plus<float>());
    // transform(vecTemp.begin(), vecTemp.end(), vecy.begin(),m_vecDirection.begin(), std::minus<float>());
}
  