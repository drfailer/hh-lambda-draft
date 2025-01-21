#ifndef HEDGEHOG_LAMBDA_TASK_H
#define HEDGEHOG_LAMBDA_TASK_H

#include <memory>
#include <string>
#include <ostream>
#include <utility>

#include <hedgehog/hedgehog.h>
#include "lambda_core_task.h"
#include "lambda_tools.h"

namespace hh {

template<size_t Separator, typename ...AllTypes>
class LambdaTask
    : public behavior::TaskNode,
      public behavior::CanTerminate,
      public behavior::Cleanable,
      public behavior::Copyable<LambdaTask<Separator, AllTypes...>>,
      public tool::LambdaTaskHelper<LambdaTask<Separator, AllTypes...>, tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorMultiReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
 private:
  std::shared_ptr<hh::core::LambdaCoreTask<Separator, AllTypes...>> const coreTask_ = nullptr; ///< Task core
  using Lambdas = tool::LambdaContainerDeducer_t<LambdaTask<Separator, AllTypes...>, tool::Inputs<Separator, AllTypes...>>;
  Lambdas lambdas_;

 public:
  using tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>::addResult;
  using behavior::TaskNode::getManagedMemory;

 public:
  explicit LambdaTask(std::string const &name, Lambdas lambdas, size_t const numberThreads = 1, bool const automaticStart = false)
      : behavior::TaskNode(std::make_shared<core::LambdaCoreTask<Separator, AllTypes...>>(this,
                                                                                    name,
                                                                                    numberThreads,
                                                                                    automaticStart)),
        behavior::Copyable<LambdaTask<Separator, AllTypes...>>(numberThreads),
        tool::LambdaTaskHelper<LambdaTask<Separator, AllTypes...>,
                               tool::Inputs<Separator, AllTypes...>>(lambdas, this),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            (std::dynamic_pointer_cast<hh::core::LambdaCoreTask<Separator, AllTypes...>>(this->core()))),
        coreTask_(std::dynamic_pointer_cast<core::LambdaCoreTask<Separator, AllTypes...>>(this->core()))
  {
    if (numberThreads == 0) { throw std::runtime_error("A task needs at least one thread."); }
    if (coreTask_ == nullptr) { throw std::runtime_error("The core used by the task should be a CoreTask."); }
  }

  explicit LambdaTask(std::shared_ptr<hh::core::LambdaCoreTask<Separator, AllTypes...>> coreTask)
      : behavior::TaskNode(std::move(coreTask)),
        behavior::Copyable<LambdaTask<Separator, AllTypes...>>(
            std::dynamic_pointer_cast<core::LambdaCoreTask<Separator, AllTypes...>>(this->core())->numberThreads()),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            std::dynamic_pointer_cast<hh::core::LambdaCoreTask<Separator, AllTypes...>>(this->core())),
        tool::LambdaTaskHelper<LambdaTask<Separator, AllTypes...>,
                               tool::Inputs<Separator, AllTypes...>>(lambdas_, this),
        coreTask_(std::dynamic_pointer_cast<core::LambdaCoreTask<Separator, AllTypes...>>(this->core()))
  { }

  ~LambdaTask() override = default;

  [[nodiscard]] size_t graphId() const { return coreTask_->graphId(); }

  [[nodiscard]] bool canTerminate() const override {
    return !coreTask_->hasNotifierConnected() && coreTask_->receiversEmpty();
  }

  [[nodiscard]] bool automaticStart() const { return this->coreTask()->automaticStart(); }

  std::shared_ptr<LambdaTask<Separator, AllTypes...>>
  copy() override {
    return std::make_shared<LambdaTask<Separator, AllTypes...>>(
        this->name(), this->lambdas_, this->numberThreads(), this->automaticStart());
  }

 protected:
  std::shared_ptr<hh::core::LambdaCoreTask<Separator, AllTypes...>> const &coreTask() const { return coreTask_; }

 public:
  [[nodiscard]] int deviceId() const { return coreTask_->deviceId(); }
};
}

#endif //HEDGEHOG_LAMBDA_TASK_H
