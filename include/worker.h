#ifndef DIST_WORKER_H
#define DIST_WORKER_H
#include<vector>
#include<ps/ps.h>

#include "adam.h"
#include "sparse_data_iter.h"
#include "sparse_sample.h"

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
        void train(SparseDataIter& iter, int num_iter,int batch_size);
        void test(SparseDataIter& iter, int num_iter);
        void setL1(bool isl1);
        void firstTrain(SparseDataIter& iter,int batch_size);
        
    private:
        float sigmoid(std::vector<Feature>& vecFeature,std::vector<float>& vecWeight);
        std::vector<float> computeGradient(std::vector<SparseSample>& batch,std::vector<float>& vecWeight);
        void computeSY(std::vector<SparseSample>& batch,int nIter);
        void pushParam();
        void pullParam();
        
        float predict(std::vector<Feature>& vecfeatures);
    public: 
        bool vectorAllzero(std::vector<float>& vec);
        
        void computePesudoGrad(std::vector<float>& vecWeight,std::vector<float>& vecGrad);
        void fixSign(std::vector<float>& vec,std::vector<float>& vecOrth);
        std::vector<float> getorthat(std::vector<float>& vecWeight);
 
    private:
        Adam* m_Adam;
        ps::KVWorker<float>* m_kv;
        int m_nfeature;
        float m_elamda;
        
        float m_l1co;
        bool m_bisL1;
        
        int m_nIteration;
        
        std::vector<float> m_vecWeightBefore;
        std::vector<float> m_vecWeightAfter;
        std::vector<float> m_vecGrad;
        std::vector<float> m_vecD;
        std::vector<float> m_vecS;
        std::vector<float> m_vecY;
        std::vector<float> m_vecLr; 
        std::vector<float> m_vecPGrad;
};
#endif 
