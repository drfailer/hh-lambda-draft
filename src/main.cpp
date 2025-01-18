#include "utils.hpp"
#include <hedgehog/hedgehog.h>

template <typename Input, typename ComputeTaskType>
using FnType = void (*)(std::shared_ptr<Input>, ComputeTaskType);

/******************************************************************************/
/*                            compute task single                             */
/******************************************************************************/

template <typename CoreTaskType, typename Input, typename... Outs>
class ComputeTaskSingle : public hh::AbstractTask<1, Input, Outs...> {
public:
  using Fn = FnType<Input, ComputeTaskSingle<CoreTaskType, Input, Outs...> *>;

  explicit ComputeTaskSingle(Fn                            pExecute,
                             std::shared_ptr<CoreTaskType> coreTask)
      : hh::AbstractTask<1, Input, Outs...>(), pExecute_(pExecute),
        coreTask_(coreTask) {}

  void execute(std::shared_ptr<Input> data) override {
    // we can't do that because all the methods are protected and we can use
    // friend in ComputeTask:
    //     pExecute_(data, computeTask_);
    pExecute_(data, this);
  }

  std::shared_ptr<hh::AbstractTask<1, Input, Outs...>> copy() override {
    return std::make_shared<ComputeTaskSingle<CoreTaskType, Input, Outs...>>(
        pExecute_, coreTask_);
  }

  // Well...
  friend Fn;
  using hh::AbstractTask<1, Input, Outs...>::addResult;
  using hh::AbstractTask<1, Input, Outs...>::getManagedMemory;
  using hh::AbstractTask<1, Input, Outs...>::deviceId;

private:
  Fn                            pExecute_;
  std::shared_ptr<CoreTaskType> coreTask_;
};

/******************************************************************************/
/*                                compute task                                */
/******************************************************************************/

template <class Ins, class Outs> class ComputeTask;

template <class... Ins, class... Outs>
class ComputeTask<In<Ins...>, Out<Outs...>>
    : public ComputeTaskSingle<hh::core::CoreTask<1, Ins, Outs...>, Ins,
                               Outs...>... {
private:
  using core_t = hh::core::CoreTask<sizeof...(Ins), Ins..., Outs...>;

public:
  // proposal for variadic friend for C++26
  // friend FnType<Ins, ComputeTaskSingle *>...;
  explicit ComputeTask(auto pExecute, std::shared_ptr<core_t> coreTask)
      : ComputeTaskSingle<hh::core::CoreTask<1, Ins, Outs...>, Ins, Outs...>(
            pExecute,
            std::dynamic_pointer_cast<hh::core::CoreTask<1, Ins, Outs...>>(
                coreTask))... {}
};

/******************************************************************************/
/*                           compute task interface                           */
/******************************************************************************/

template <typename Input, typename Output> class ComputeTaskInterface;

template <typename... Ins, typename... Outs>
class ComputeTaskInterface<In<Ins...>, Out<Outs...>>
    : public ComputeTask<In<Ins...>, Out<Outs...>> {
private:
  using core_t = hh::core::CoreTask<sizeof...(Ins), Ins..., Outs...>;
  using abstract_task_t = hh::AbstractTask<sizeof...(Ins), Ins..., Outs...>;

public:
  ComputeTaskInterface(const std::string name, auto pExecute,
                       int32_t numberThreads = 1, bool automaticStart = false)
      : ComputeTask<In<Ins...>, Out<Outs...>>(
            pExecute,
            std::make_shared<core_t>(dynamic_cast<abstract_task_t *>(this),
                                     name, numberThreads, automaticStart)) {}
};

/******************************************************************************/
/*                                    main                                    */
/******************************************************************************/

int main() {
  auto Simple =
      std::make_shared<ComputeTaskInterface<In<int, double>, Out<int>>>(
          "Simple", []<typename T>(std::shared_ptr<T> data, auto *pThis) {
            if constexpr (std::is_same_v<T, int>) {
              (*data)++;
              printf("[Simple][Data %d][Device ID %d]\n", *data,
                     pThis->deviceId());
              pThis->getManagedMemory();
              pThis->addResult(data);
            } else if constexpr (std::is_same_v<T, double>) {
              (*data)++;
              printf("[Simple][Data %f][Device ID %d]\n", *data,
                     pThis->deviceId());
              pThis->getManagedMemory();
              pThis->addResult(std::make_shared<int>(*data));
            }
          });

  auto test =
      std::dynamic_pointer_cast<hh::AbstractTask<2, int, double, int>>(Simple);

  std::cout << test << std::endl;
  return 0;
}
