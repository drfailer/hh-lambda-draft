#ifndef HEDGEHOG_INTERNAL_LAMBDA_TASK_H
#define HEDGEHOG_INTERNAL_LAMBDA_TASK_H

#include <memory>
#include <string>
#include <ostream>
#include <utility>

#include <hedgehog/hedgehog.h>
#include "lambda_core_task.h"
#include "lambda_tools.h"

namespace hh {

// this class should be abstract
template <typename SubType, size_t Separator, typename... AllTypes>
class InternalLambdaTask
    : public behavior::TaskNode,
      public behavior::CanTerminate,
      public behavior::Cleanable,
      public behavior::Copyable<InternalLambdaTask<SubType, Separator, AllTypes...>>,
      public tool::LambdaTaskHelper<
          std::conditional_t<std::is_same_v<SubType, void>,
              InternalLambdaTask<SubType, Separator, AllTypes...>, SubType>,
          tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorMultiReceiversTypeDeducer_t<
          tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorTaskMultiSendersTypeDeducer_t<
          tool::Outputs<Separator, AllTypes...>> {
private:
  std::shared_ptr<hh::core::LambdaCoreTask<SubType, Separator, AllTypes...>> const coreTask_ = nullptr; ///< Task core

  using LambdaTaskType = std::conditional_t<std::is_same_v<SubType, void>, InternalLambdaTask<SubType, Separator, AllTypes...>, SubType>;
  using Lambdas = tool::LambdaContainerDeducer_t<LambdaTaskType, tool::Inputs<Separator, AllTypes...>>;
  Lambdas lambdas_ = {};
  LambdaTaskType *self_ = nullptr;

 public:
  using tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>::addResult;
  using behavior::TaskNode::getManagedMemory;

 public:
  explicit InternalLambdaTask(std::string const &name, Lambdas lambdas, size_t const numberThreads = 1, bool const automaticStart = false, LambdaTaskType *self = nullptr)
      : behavior::TaskNode(std::make_shared<core::LambdaCoreTask<SubType, Separator, AllTypes...>>(this,
                                                                                    name,
                                                                                    numberThreads,
                                                                                    automaticStart)),
        behavior::Copyable<InternalLambdaTask<SubType, Separator, AllTypes...>>(numberThreads),
        tool::LambdaTaskHelper<LambdaTaskType, tool::Inputs<Separator, AllTypes...>>(lambdas, self),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            (std::dynamic_pointer_cast<hh::core::LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core()))),
        coreTask_(std::dynamic_pointer_cast<core::LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())),
        lambdas_(lambdas),
        self_(self)
  {
    if (numberThreads == 0) { throw std::runtime_error("A task needs at least one thread."); }
    if (coreTask_ == nullptr) { throw std::runtime_error("The core used by the task should be a CoreTask."); }
  }

  explicit InternalLambdaTask(std::string const &name, size_t const numberThreads = 1, bool const automaticStart = false, LambdaTaskType *self = nullptr)
      : InternalLambdaTask<SubType, Separator, AllTypes...>(name, {}, numberThreads, automaticStart, self) { }

  // dangerous
  explicit InternalLambdaTask(std::shared_ptr<hh::core::LambdaCoreTask<SubType, Separator, AllTypes...>> coreTask, LambdaTaskType *self)
      : behavior::TaskNode(std::move(coreTask)),
        behavior::Copyable<InternalLambdaTask<SubType, Separator, AllTypes...>>(
            std::dynamic_pointer_cast<core::LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())->numberThreads()),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            std::dynamic_pointer_cast<hh::core::LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())),
        tool::LambdaTaskHelper<InternalLambdaTask<SubType, Separator, AllTypes...>,
                               tool::Inputs<Separator, AllTypes...>>({}, self),
        coreTask_(std::dynamic_pointer_cast<core::LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())),
        lambdas_({}),
        self_(self)
  { }

  ~InternalLambdaTask() override = default;

  template<hh::tool::ContainsInTupleConcept<tool::Inputs<Separator, AllTypes...>> Input>
  void setLambda(void(lambda)(std::shared_ptr<Input>, LambdaTaskType*)) {
      std::get<void(*)(std::shared_ptr<Input>, LambdaTaskType*)>(lambdas_) = lambda;
      tool::LambdaTaskHelper<LambdaTaskType, tool::Inputs<Separator, AllTypes...>>::reinitialize(lambdas_, self_);
  }

  [[nodiscard]] size_t graphId() const { return coreTask_->graphId(); }

  [[nodiscard]] bool canTerminate() const override {
    return !coreTask_->hasNotifierConnected() && coreTask_->receiversEmpty();
  }

  [[nodiscard]] bool automaticStart() const { return this->coreTask()->automaticStart(); }

  std::shared_ptr<InternalLambdaTask<SubType, Separator, AllTypes...>>
  copy() override {
    return std::make_shared<InternalLambdaTask<SubType, Separator, AllTypes...>>(
        this->name(), this->lambdas_, this->numberThreads(), this->automaticStart());
  }

 protected:
  std::shared_ptr<hh::core::LambdaCoreTask<SubType, Separator, AllTypes...>> const &coreTask() const { return coreTask_; }

 public:
  [[nodiscard]] int deviceId() const { return coreTask_->deviceId(); }
};
}

#endif //HEDGEHOG_LAMBDA_TASK_H
