#include "hh/lambda_task.h"
#include <hedgehog/hedgehog.h>

struct CPUTask: public hh::AbstractTask<1, int, float>{};

// Usecase: CUDA Task, cublas Task...
template<size_t Separator, typename ...AllTypes>
class MySpecializedLambdaTask
    : public hh::LambdaTask<MySpecializedLambdaTask<Separator, AllTypes...>, Separator, AllTypes...> {
public:
    explicit MySpecializedLambdaTask(const std::string name, size_t numberThreads = 1, bool automaticStart = false):
        hh::LambdaTask<MySpecializedLambdaTask<Separator, AllTypes...>, Separator, AllTypes...>(name, this, numberThreads, automaticStart) {}

    void initialize() override {
        myHandle_ = 42;
    }

    void shutdown() override {
        myHandle_ = -1;
    }

    [[nodiscard]] int32_t getMyHandle() {
        return myHandle_;
    }
private:
    int32_t myHandle_ = 0;//cudaStream_t, cublasHandle_t
};

int main() {
//  auto Simple = std::make_shared<hh::LambdaTask<2, int, double, int>>(
//      "Simple", std::tuple{
//                    [](std::shared_ptr<int> data, auto *pThis) {
//                      (*data)++;
//                      printf("[Simple<int>][Data %d][Device ID %d]\n", *data,
//                             pThis->deviceId());
//                      /* pThis->getManagedMemory(); */
//                      pThis->addResult(data);
//                    },
//                    [](std::shared_ptr<double> data, auto *pThis) {
//                      (*data)++;
//                      printf("[Simple<double>][Data %f][Device ID %d]\n", *data,
//                             pThis->deviceId());
//                      /* pThis->getManagedMemory(); */
//                      pThis->addResult(std::make_shared<int>(*data));
//                    },
//                });

    auto Simple = std::make_shared<MySpecializedLambdaTask<2, int, double, int>>("Special");
    Simple->setLambda<int>([](std::shared_ptr<int> data, MySpecializedLambdaTask<2, int, double, int> *pThis) {
        (*data)++;
        /* pThis->getManagedMemory(); */
        pThis->addResult(data);
        printf("[Simple<int>][Data %d][Device ID %d][MyHandle %d]\n", *data, pThis->deviceId(), pThis->getMyHandle());
    });

    Simple->setLambda<double>([](std::shared_ptr<double> data, MySpecializedLambdaTask<2, int, double, int> *pThis) {
        (*data)++;
        /* pThis->getManagedMemory(); */
        pThis->addResult(std::make_shared<int>(*data));
        printf("[Simple<double>][Data %f][Device ID %d][MyHandle %d]\n", *data, pThis->deviceId(), pThis->getMyHandle());
    });

  hh::Graph<2, int, double, int> graph("test");

  graph.inputs(Simple);
  graph.outputs(Simple);

  graph.executeGraph();
  graph.pushData(std::make_shared<int>(1));
  graph.pushData(std::make_shared<double>(2.3));
  graph.finishPushingData();
  graph.waitForTermination();
  return 0;
}
