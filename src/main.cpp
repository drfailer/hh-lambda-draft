#include "utils.hpp"
#include <hedgehog/hedgehog.h>

/******************************************************************************/
/*                            compute task single                             */
/******************************************************************************/

template <typename CoreTaskType, typename Input, typename... Outputs>
class SingleInputTask : public hh::AbstractTask<1, Input, Outputs...> {
public:
  using Fn = FnType<Input, SingleInputTask<CoreTaskType, Input, Outputs...> *>;

  explicit SingleInputTask(Fn                            executable,
                           std::shared_ptr<CoreTaskType> coreTask)
      : hh::AbstractTask<1, Input, Outputs...>(coreTask),
        executable_(executable), coreTask_(coreTask) {}

  void execute(std::shared_ptr<Input> data) override {
    // we can't do that because all the methods are protected and we can use
    // friend in ComputeTask:
    //     executable_(data, computeTask_);
    executable_(data, this);
  }

  std::shared_ptr<hh::AbstractTask<1, Input, Outputs...>> copy() override {
    return std::make_shared<SingleInputTask<CoreTaskType, Input, Outputs...>>(
        executable_, coreTask_);
  }

  // For a reason, friendship only didn't work
  friend Fn;
  using hh::AbstractTask<1, Input, Outputs...>::addResult;
  using hh::AbstractTask<1, Input, Outputs...>::getManagedMemory;
  using hh::AbstractTask<1, Input, Outputs...>::deviceId;

private:
  Fn                            executable_;
  std::shared_ptr<CoreTaskType> coreTask_;
};

/******************************************************************************/
/*                                compute task                                */
/******************************************************************************/

template <class Inputs, class Outputs> class SingleInputTaskCombinator;

template <class... Inputs, class... Outputs>
class SingleInputTaskCombinator<std::tuple<Inputs...>, std::tuple<Outputs...>>
    : public SingleInputTask<hh::core::CoreTask<1, Inputs, Outputs...>, Inputs,
                             Outputs...>... {
private:
  using CoreTaskType =
      hh::core::CoreTask<sizeof...(Inputs), Inputs..., Outputs...>;

public:
  // proposal for variadic friend for C++26
  // friend FnType<Inputs, ComputeTaskSingle *>...;

  template <typename Executable>
  explicit SingleInputTaskCombinator(Executable                    executable,
                                     std::shared_ptr<CoreTaskType> coreTask)
      : SingleInputTask<hh::core::CoreTask<1, Inputs, Outputs...>, Inputs,
                        Outputs...>(
            executable,
            std::dynamic_pointer_cast<
                hh::core::CoreTask<1, Inputs, Outputs...>>(coreTask))... {}
};

/******************************************************************************/
/*                           compute task interface                           */
/******************************************************************************/

template <size_t Separator, typename... Types>
class LambdaTask
    : public SingleInputTaskCombinator<hh::tool::Inputs<Separator, Types...>,
                                       hh::tool::Outputs<Separator, Types...>> {
private:
  using CoreTaskType = hh::core::CoreTask<Separator, Types...>;
  using AbstractTaskType = hh::AbstractTask<Separator, Types...>;

public:
  template <typename Executable>
  LambdaTask(const std::string name, Executable executable,
             int32_t numberThreads = 1, bool automaticStart = false)
      : SingleInputTaskCombinator<hh::tool::Inputs<Separator, Types...>,
                                  hh::tool::Outputs<Separator, Types...>>(
            executable, std::make_shared<CoreTaskType>(
                            dynamic_cast<AbstractTaskType *>(this), name,
                            numberThreads, automaticStart)) {}
};

/******************************************************************************/
/*                                    main                                    */
/******************************************************************************/

int main() {
  auto Simple = std::make_shared<LambdaTask<2, int, double, int>>(
      "Simple", []<typename T>(std::shared_ptr<T> data, auto *pThis) {
        if constexpr (std::is_same_v<T, int>) {
          (*data)++;
          printf("[Simple][Data %d][Device ID %d]\n", *data, pThis->deviceId());
          pThis->getManagedMemory();
          pThis->addResult(data);
        } else if constexpr (std::is_same_v<T, double>) {
          (*data)++;
          printf("[Simple][Data %f][Device ID %d]\n", *data, pThis->deviceId());
          pThis->getManagedMemory();
          pThis->addResult(std::make_shared<int>(*data));
        }
      });
  return 0;
}
