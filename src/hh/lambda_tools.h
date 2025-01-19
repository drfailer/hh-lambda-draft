#ifndef HEDGEHOG_LAMBDA_TOOLS_H
#define HEDGEHOG_LAMBDA_TOOLS_H

#include <hedgehog/hedgehog.h>

namespace hh {

// helper to implement the execute function ////////////////////////////////////

// TODO: find a beter name for these and move in separated files

namespace tool {

template <typename LambdaType, typename LambdaTaskType, typename Input>
class SingleInputTask
    : public BehaviorMultiExecuteTypeDeducer_t<std::tuple<Input>>
{
  private:
    LambdaType lambda_;
    LambdaTaskType *task_;

  public:
    SingleInputTask(LambdaType lambda, LambdaTaskType *task)
        : lambda_(lambda), task_(task) { }

    void execute(std::shared_ptr<Input> data) override {
        lambda_(data, task_);
    }
};

template<typename LambdaType, typename LambdaTaskType, typename Input>
class LambdaTaskHelper;

template<typename LambdaType, typename LambdaTaskType, typename ...Inputs>
class LambdaTaskHelper<LambdaType, LambdaTaskType, std::tuple<Inputs...>>
    : public SingleInputTask<LambdaType, LambdaTaskType, Inputs>... {
  public:
      LambdaTaskHelper(LambdaType lambda, LambdaTaskType *task)
          : SingleInputTask<LambdaType, LambdaTaskType, Inputs>(lambda, task)... {}
};

}

}

#endif
