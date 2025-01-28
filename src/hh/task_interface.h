#ifndef HEDGEHOG_TASK_INTERFACE_H
#define HEDGEHOG_TASK_INTERFACE_H
#include "hedgehog/src/api/memory_manager/managed_memory.h"
#include <memory>

namespace hh {

namespace tool {

template <typename TaskType>
class TaskInterface {
  private:
    TaskType *task_ = nullptr;

  public:
    TaskInterface(TaskType *task) : task_(task) {}

    ~TaskInterface() = default;

  public:
    template <typename Type>
    void addResult(std::shared_ptr<Type> data) {
        task_->addResult(data);
    }

    std::shared_ptr<ManagedMemory> getManagedMemory() {
        return task_->getManagedMemory();
    }

    auto const &coreTask() const { return task_->coreTask_; }

    [[nodiscard]] int deviceId() const { return task_->coreTask_->deviceId(); }

    TaskType *operator->() const { return task_; }

    void task(TaskType *task) { this->task_ = task; }
};

}

}

#endif
