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
        
    private:
        float sigmoid(std::vector<Feature>& vecFeature,std::vector<float>& vecWeight);
        std::vector<float> computeGradient(std::vector<SparseSample>& batch,std::vector<float>& vecWeight);
        void computeSY(std::vector<SparseSample>& batch,int nIter);
        void pushParam();
        void pullParam();
        
        float predict(std::vector<Feature>& vecfeatures);
        
        bool vectorAllzero(std::vector<float>& vec);
        std::vector<float> convertSparseToDense(std::vector<Feature>& vecSparse);
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
