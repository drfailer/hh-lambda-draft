#include "hh/lambda_task.h"
#include <hedgehog/hedgehog.h>

int main() {
  auto Simple = std::make_shared<hh::LambdaTask<2, int, double, int>>(
      "Simple", std::tuple{
                    [](std::shared_ptr<int> data, auto *pThis) {
                      (*data)++;
                      printf("[Simple<int>][Data %d][Device ID %d]\n", *data,
                             pThis->deviceId());
                      /* pThis->getManagedMemory(); */
                      pThis->addResult(data);
                    },
                    [](std::shared_ptr<double> data, auto *pThis) {
                      (*data)++;
                      printf("[Simple<double>][Data %f][Device ID %d]\n", *data,
                             pThis->deviceId());
                      /* pThis->getManagedMemory(); */
                      pThis->addResult(std::make_shared<int>(*data));
                    },
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
