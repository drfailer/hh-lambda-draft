#include "hh/lambda_task.h"
#include <hedgehog/hedgehog.h>

/******************************************************************************/
/*                                    main                                    */
/******************************************************************************/

int main() {
  auto f =
      []<typename T>(std::shared_ptr<T> data, auto *pThis) {
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
      };

  auto Simple = std::make_shared<hh::LambdaTask<decltype(f), 2, int, double, int>>(
      "Simple", f);

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
