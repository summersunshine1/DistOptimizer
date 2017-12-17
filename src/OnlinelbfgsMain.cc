
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
    worker.setKVWorker(kv);
    
    int num_iteration = distlr::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));
    int batch_size = distlr::ToInt(ps::Environment::Get()->find("BATCH_SIZE"));
    int test_interval = distlr::ToInt(ps::Environment::Get()->find("TEST_INTERVAL"));
    if(rank==0)
    {
        std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
        SparseDataIter iter(filename);
        worker.train(iter, 1, batch_size);
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
    return 0;
}
