
#include "worker.h"
#include "distserver.h"

void StartServer() {
    if (!ps::IsServer()) {
        return;
    }
    float lamda = distlr::ToFloat(ps::Environment::Get()->find("LAMDA"));
    DistServer* pserver = new DistServer(lamda);
    ps::RegisterExitCallback([pserver](){ delete pserver; });
}

void RunWorker() {
    if (!ps::IsWorker()) {
        return;
    }

    std::string root = ps::Environment::Get()->find("DATA_DIR");
    int num_feature_dim = distlr::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));
    float lamda = distlr::ToFloat(ps::Environment::Get()->find("LAMDA"));

    int rank = ps::MyRank();
    ps::KVWorker<float>* kv = new ps::KVWorker<float>(0);
    Worker worker(num_feature_dim,lamda);
    bool isl1 = distlr::ToInt(ps::Environment::Get()->find("L1")) ? true:false;
    worker.setKVWorker(kv);
    worker.setL1(isl1);
    
    int num_iteration = distlr::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));
    int batch_size = distlr::ToInt(ps::Environment::Get()->find("BATCH_SIZE"));
    int test_interval = distlr::ToInt(ps::Environment::Get()->find("TEST_INTERVAL"));
    
    if(rank==0)
    {
        std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
        SparseDataIter iter(filename);
        worker.firstTrain(iter, batch_size);
    }
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
    std::cout << "Worker[" << rank << "]: start working..." << std::endl;
    for (int i = 0; i < num_iteration; ++i) {
        std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
        SparseDataIter iter(filename);
        worker.train(iter, i+1, batch_size);
        // if (rank==0 && (i + 1) % test_interval == 0) {
            // std::cout<<"test...."<<rank<<std::endl;
            // std::string filename = root + "/test/part-001";
            // SparseDataIter test_iter(filename);
            // worker.test(test_iter, i+1);
        // }
    }
}
int main() {
    StartServer();
    ps::Start();

    RunWorker();

    ps::Finalize();
    // Worker worker(10,0.1);
    // float b[]={2,2,-1,-3,0};
    // float a[]={1,-1,2,1,0};
    // std::vector<float> v(b,b+sizeof(b)/sizeof(float));
    // std::vector<float> v1(a,a+sizeof(a)/sizeof(float));
    
    // worker.fixSign(v,v1);
    // for(int i=0;i<v.size();i++)
    // {
        // std::cout<<v[i]<<" ";
    // }
    // std::cout<<std::endl;
    return 0;
}
