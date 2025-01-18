#ifndef SRC_UTILS_H
#define SRC_UTILS_H
#include <cstddef>
#include <variant>

template <typename... Types> using Wrapper = std::variant<Types...>;
template <typename... Types> struct In {};
template <typename... Types> struct Out {};

template <typename T>
struct size;

template <typename ...Types>
struct size<In<Types...>>{
    constexpr static size_t value = sizeof...(Types);
};

template <typename T>
constexpr size_t size_v = size<T>::value;

#endif
