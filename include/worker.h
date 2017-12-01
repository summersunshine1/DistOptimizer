#ifndef SERVER_H_
#define SERVER_H_
#include<vector>
#include<ps/ps.h>

#include "adam.h"
#include "data_iter.h"
#include "sample.h"

using namespace distlr;
class Worker
{
    public:
        Worker(int n_feature,float n_lamda);
        void setKVWorker(ps::KVWorker<float>* kv)
        {
            m_kv = kv;
        }
        ~Worker()
        {
            if (m_kv) {
                delete m_kv;
            }
            if (m_Adam) {
                delete m_Adam;
            }
        }
        void train(DataIter& iter, int num_iter,int batch_size);
        void test(DataIter& iter, int num_iter);
        
    private:
        float sigmoid(std::vector<float>& vecFeature,std::vector<float>& vecWeight);
        std::vector<float> computeGradient(std::vector<Sample>& batch,std::vector<float>& vecWeight);
        void computeSY(std::vector<Sample>& batch,int nIter);
        void pushParam();
        void pullParam();
        
        int predict(std::vector<float>& vecfeatures);
        
        bool vectorAllzero(std::vector<float>& vec);

    private:
        Adam* m_Adam;
        ps::KVWorker<float>* m_kv;
        int m_nfeature;
        float m_elamda;
        
        std::vector<float> m_vecWeightBefore;
        std::vector<float> m_vecWeightAfter;
        std::vector<float> m_vecGrad;
        std::vector<float> m_vecD;
        std::vector<float> m_vecS;
        std::vector<float> m_vecY;
        std::vector<float> m_vecLr;    
};
#endif 
