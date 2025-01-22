#ifndef HEDGEHOG_LAMBDA_TOOLS_H
#define HEDGEHOG_LAMBDA_TOOLS_H

#include <hedgehog/hedgehog.h>

namespace hh {

// helper to implement the execute function ////////////////////////////////////

// TODO: find a beter name for these and move in separated files

namespace tool {

template <typename LambdaTaskType, typename Input>
class SingleInputTask
    : public BehaviorMultiExecuteTypeDeducer_t<std::tuple<Input>>
{
  private:
    using LambdaType = void(*)(std::shared_ptr<Input>, LambdaTaskType*);
    LambdaType lambda_;
    LambdaTaskType *task_;

  public:
    SingleInputTask(LambdaType lambda, LambdaTaskType *task)
        : lambda_(lambda), task_(task) { }

    void execute(std::shared_ptr<Input> data) override {
        lambda_(data, task_);
    }

    void reinitialize(LambdaType lambda, LambdaTaskType *task) {
        lambda_ = lambda;
        task_   = task;
    }
};

template<typename LambdaTaskType, typename Input>
class LambdaTaskHelper;

template<typename LambdaTaskType, typename ...Inputs>
class LambdaTaskHelper<LambdaTaskType, std::tuple<Inputs...>>
    : public SingleInputTask<LambdaTaskType, Inputs>... {
  public:
      using LambdaContainer = std::tuple<void(*)(std::shared_ptr<Inputs>, LambdaTaskType*)...>;

  public:
      LambdaTaskHelper(LambdaContainer lambdas, LambdaTaskType *task)
          : SingleInputTask<LambdaTaskType, Inputs>(
                  std::get<void(*)(std::shared_ptr<Inputs>, LambdaTaskType*)>(lambdas), task)... {}

      void reinitialize(LambdaContainer lambdas, LambdaTaskType *task) {
          (SingleInputTask<LambdaTaskType, Inputs>::reinitialize(std::get<void(*)(std::shared_ptr<Inputs>, LambdaTaskType*)>(lambdas), task), ...);
      }
};


template <typename LambdaTaskType, typename ...Inputs>
using LambdaContainer = std::tuple<void(*)(std::shared_ptr<Inputs>, LambdaTaskType*)...>;

template<class LambdaTaskType, class Inputs>
struct LambdaContainerDeducer;

template<class LambdaTaskType, class ...Inputs>
struct LambdaContainerDeducer<LambdaTaskType, std::tuple<Inputs...>> { using type = LambdaContainer<LambdaTaskType, Inputs...>; };

template<class LambdaTaskType, class TupleInputs>
using LambdaContainerDeducer_t = typename LambdaContainerDeducer<LambdaTaskType, TupleInputs>::type;

}

}

#endif
