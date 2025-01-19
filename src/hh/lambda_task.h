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

template<typename LambdaType, size_t Separator, typename ...AllTypes>
class LambdaTask
    : public behavior::TaskNode,
      public behavior::CanTerminate,
      public behavior::Cleanable,
      public behavior::Copyable<LambdaTask<LambdaType, Separator, AllTypes...>>,
      public tool::LambdaTaskHelper<LambdaType, LambdaTask<LambdaType, Separator, AllTypes...>, tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorMultiReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
 private:
  std::shared_ptr<hh::core::LambdaCoreTask<LambdaType, Separator, AllTypes...>> const coreTask_ = nullptr; ///< Task core
  LambdaType lambda_;

 public:
  friend LambdaType; // friendship doesn't work because the lambda is template I
                     // think (for me it looks like a bug but it may also be
                     // specified in the standard).
  using tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>::addResult;
  using behavior::TaskNode::getManagedMemory;

 public:
  explicit LambdaTask(std::string const &name, LambdaType lambda, size_t const numberThreads = 1, bool const automaticStart = false)
      : behavior::TaskNode(std::make_shared<core::LambdaCoreTask<LambdaType, Separator, AllTypes...>>(this,
                                                                                    name,
                                                                                    numberThreads,
                                                                                    automaticStart)),
        behavior::Copyable<LambdaTask<LambdaType, Separator, AllTypes...>>(numberThreads),
        tool::LambdaTaskHelper<LambdaType, LambdaTask<LambdaType, Separator, AllTypes...>,
                               tool::Inputs<Separator, AllTypes...>>(lambda, this),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            (std::dynamic_pointer_cast<hh::core::LambdaCoreTask<LambdaType, Separator, AllTypes...>>(this->core()))),
        coreTask_(std::dynamic_pointer_cast<core::LambdaCoreTask<LambdaType, Separator, AllTypes...>>(this->core()))
  {
    if (numberThreads == 0) { throw std::runtime_error("A task needs at least one thread."); }
    if (coreTask_ == nullptr) { throw std::runtime_error("The core used by the task should be a CoreTask."); }
  }

  explicit LambdaTask(std::shared_ptr<hh::core::LambdaCoreTask<LambdaType, Separator, AllTypes...>> coreTask)
      : behavior::TaskNode(std::move(coreTask)),
        behavior::Copyable<LambdaTask<LambdaType, Separator, AllTypes...>>(
            std::dynamic_pointer_cast<core::LambdaCoreTask<LambdaType, Separator, AllTypes...>>(this->core())->numberThreads()),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            std::dynamic_pointer_cast<hh::core::LambdaCoreTask<LambdaType, Separator, AllTypes...>>(this->core())),
        tool::LambdaTaskHelper<LambdaType, LambdaTask<LambdaType, Separator, AllTypes...>,
                               tool::Inputs<Separator, AllTypes...>>(lambda_, this),
        coreTask_(std::dynamic_pointer_cast<core::LambdaCoreTask<LambdaType, Separator, AllTypes...>>(this->core()))
  { }

  ~LambdaTask() override = default;

  [[nodiscard]] size_t graphId() const { return coreTask_->graphId(); }

  [[nodiscard]] bool canTerminate() const override {
    return !coreTask_->hasNotifierConnected() && coreTask_->receiversEmpty();
  }

  [[nodiscard]] bool automaticStart() const { return this->coreTask()->automaticStart(); }

  std::shared_ptr<LambdaTask<LambdaType, Separator, AllTypes...>>
  copy() override {
    return std::make_shared<LambdaTask<LambdaType, Separator, AllTypes...>>(
        this->name(), this->lambda_, this->numberThreads(), this->automaticStart());
  }

 protected:
  std::shared_ptr<hh::core::LambdaCoreTask<LambdaType, Separator, AllTypes...>> const &coreTask() const { return coreTask_; }

 public:
  // WARN: public for sharing with the lambda
  [[nodiscard]] int deviceId() const { return coreTask_->deviceId(); }
};
}

#endif //HEDGEHOG_LAMBDA_TASK_H
