#ifndef DISTSERVER_H_
#define DISTSERVER_H_
#include<vector>
#include<ps/ps.h>

class DistServer
{
    public:
        DistServer(float lamda);
        ~DistServer() 
        {
            if (m_psServer) {
                delete m_psServer;
            }
        }
    private:
        void updateSY();
        void dataHandle(const ps::KVMeta& req_meta,const ps::KVPairs<float>& req_data,ps::KVServer<float>* server); 
        void init();
        void pushSY(std::vector<float>& vecTemps, std::vector<float>& vecTempy);
        int DecodeKey(ps::Key key);
        void initFirstDirection(std::vector<float>& vecGrad, std::vector<float>& vecs, std::vector<float>& vecy);
        void pop_front(std::vector<std::vector<float>>& vec);
    private:
        std::vector<std::vector<float>> m_vecS;
        std::vector<std::vector<float>> m_vecY;
        std::vector<float> m_vecGrad;
        std::vector<float> m_vecWeight;
        std::vector<float> m_vecDirection;
        
        int m_nWindow;
        int m_nFeature;
        float m_eEpisilo;
        float m_eScale;
        float m_eLamda;
        
        bool m_bSync;
        ps::KVServer<float>* m_psServer;
};
#endif 