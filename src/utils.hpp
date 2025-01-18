#ifndef SRC_UTILS_H
#define SRC_UTILS_H
#include <memory>

template <typename... Types> struct In {};
template <typename... Types> struct Out {};

template <typename Input, typename ThisType>
using FnType = void (*)(std::shared_ptr<Input>, ThisType);

#endif
