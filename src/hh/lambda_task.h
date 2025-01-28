#ifndef HEDGEHOG_LAMBDA_TASK_H
#define HEDGEHOG_LAMBDA_TASK_H

#include "internal_lambda_task.h"

namespace hh {

template <typename SubType, size_t Separator, typename ...AllTypes>
struct LambdaTask : InternalLambdaTask<SubType, Separator, AllTypes...> {
  using Lambdas = tool::LambdaContainerDeducer_t<SubType, tool::Inputs<Separator, AllTypes...>>;

  explicit LambdaTask(std::string const &name, Lambdas lambdas, SubType *self,
                      size_t const numberThreads = 1,
                      bool const automaticStart = false)
      : InternalLambdaTask<SubType, Separator, AllTypes...>(
            name, lambdas, numberThreads, automaticStart, self) {}

  explicit LambdaTask(std::string const &name, SubType *self,
                      size_t const numberThreads = 1,
                      bool const automaticStart = false)
      : LambdaTask<SubType, Separator, AllTypes...>(name, {}, self, numberThreads,
                                                    automaticStart) {}
};

template <size_t Separator, typename ...AllTypes>
struct LambdaTask<void, Separator, AllTypes...>
    : InternalLambdaTask<void, Separator, AllTypes...> {
  using Lambdas = tool::LambdaContainerDeducer_t<InternalLambdaTask<void, Separator, AllTypes...>, tool::Inputs<Separator, AllTypes...>>;

  explicit LambdaTask(std::string const &name, Lambdas lambdas,
                      size_t const numberThreads = 1,
                      bool const automaticStart = false)
      : InternalLambdaTask<InternalLambdaTask<void, Separator, AllTypes...>,
                           Separator, AllTypes...>(name, lambdas, numberThreads,
                                                   automaticStart, this) {}

  explicit LambdaTask(std::string const &name, size_t const numberThreads = 1,
                      bool const automaticStart = false)
      : LambdaTask<void, Separator, AllTypes...>(name, {}, numberThreads,
                                                 automaticStart) {}
};

}

#endif
