#include "hh/lambda_task.h"
#include <hedgehog/hedgehog.h>

// this functions should be in hh v
template <size_t Separator, typename... AllTypes>
auto makeLambdaTask(std::string const &name, auto lambda,
                    size_t numberThreads = 1) {
  return std::make_shared<
      hh::LambdaTask<decltype(lambda), Separator, AllTypes...>>(name, lambda,
                                                                numberThreads);
}

int main() {
  auto Simple = makeLambdaTask<2, int, double, int>(
      "Simple", []<typename T>(std::shared_ptr<T> data, auto *pThis) {
        if constexpr (std::is_same_v<T, int>) {
          (*data)++;
          printf("[Simple][Data %d][Device ID %d]\n", *data, pThis->deviceId());
          /* pThis->getManagedMemory(); */
          pThis->addResult(data);
        } else if constexpr (std::is_same_v<T, double>) {
          (*data)++;
          printf("[Simple][Data %f][Device ID %d]\n", *data, pThis->deviceId());
          /* pThis->getManagedMemory(); */
          pThis->addResult(std::make_shared<int>(*data));
        }
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
