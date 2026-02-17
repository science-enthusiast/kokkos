// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP
#define KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP

#include <type_traits>

#include <Kokkos_Layout.hpp>
#include <Kokkos_Macros.hpp>

#if defined(__GNUC__)
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP _Pragma("GCC ivdep")
#elif defined(__clang__)
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP \
  _Pragma("clang loop vectorize(assume_safety)")
#else
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP _Pragma("ivdep")
#endif

namespace Kokkos {
namespace Impl {

// MDRangePolicy iteration via a nested loop without tiles

// ParallelFor macros: Non-Tagged

#define KOKKOS_IMPL_NLWOTILE_APPLY(func, ...) func(__VA_ARGS__);

// Non-Tagged: Outer Iterate::Right
#define KOKKOS_IMPL_NLWOTILE_R1(type, rp, func, rank, ...)   \
    KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                       \
    for (auto i0 = static_cast<type>(rp.m_lower[rank - 1]);  \
       i0 < static_cast<type>(rp.m_upper[rank - 1]); ++i0) { \
    KOKKOS_IMPL_NLWOTILE_APPLY(func, __VA_ARGS__, i0)        \
  }

#define KOKKOS_IMPL_NLWOTILE_R2(type, rp, func, rank, ...)         \
  for (auto i1 = static_cast<type>(rp.m_lower[rank - 2]);          \
       i1 < static_cast<type>(rp.m_upper[rank - 2]); ++i1) {       \
    KOKKOS_IMPL_NLWOTILE_R1(type, rp, func, rank, __VA_ARGS__, i1) \
  }

#define KOKKOS_IMPL_NLWOTILE_R3(type, rp, func, rank, ...)         \
  for (auto i2 = static_cast<type>(rp.m_lower[rank - 3]);          \
       i2 < static_cast<type>(rp.m_upper[rank - 3]); ++i2) {       \
    KOKKOS_IMPL_NLWOTILE_R2(type, rp, func, rank, __VA_ARGS__, i2) \
  }

#define KOKKOS_IMPL_NLWOTILE_R4(type, rp, func, rank, ...)         \
  for (auto i3 = static_cast<type>(rp.m_lower[rank - 4]);          \
       i3 < static_cast<type>(rp.m_upper[rank - 4]); ++i3) {       \
    KOKKOS_IMPL_NLWOTILE_R3(type, rp, func, rank, __VA_ARGS__, i3) \
  }

#define KOKKOS_IMPL_NLWOTILE_R5(type, rp, func, rank, ...)         \
  for (auto i4 = static_cast<type>(rp.m_lower[rank - 5]);          \
       i4 < static_cast<type>(rp.m_upper[rank - 5]); ++i4) {       \
    KOKKOS_IMPL_NLWOTILE_R4(type, rp, func, rank, __VA_ARGS__, i4) \
  }

#define KOKKOS_IMPL_NLWOTILE_R6(type, rp, func, rank, ...)         \
  for (auto i5 = static_cast<type>(rp.m_lower[rank - 6]);          \
       i5 < static_cast<type>(rp.m_upper[rank - 6]); ++i5) {       \
    KOKKOS_IMPL_NLWOTILE_R5(type, rp, func, rank, __VA_ARGS__, i5) \
  }

#define KOKKOS_IMPL_NLWOTILE_R7(type, rp, func, rank, ...)         \
  for (auto i6 = static_cast<type>(rp.m_lower[rank - 7]);          \
       i6 < static_cast<type>(rp.m_upper[rank - 7]); ++i6) {       \
    KOKKOS_IMPL_NLWOTILE_R6(type, rp, func, rank, __VA_ARGS__, i6) \
  }

// Non-Tagged: Outer Iterate::Left
#define KOKKOS_IMPL_NLWOTILE_L1(type, rp, func, ...)  \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                  \
  for (auto i0 = static_cast<type>(rp.m_lower[0]);    \
       i0 < static_cast<type>(rp.m_upper[0]); ++i0) { \
    KOKKOS_IMPL_NLWOTILE_APPLY(func, i0, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_NLWOTILE_L2(type, rp, func, ...)         \
  for (auto i1 = static_cast<type>(rp.m_lower[1]);           \
       i1 < static_cast<type>(rp.m_upper[1]); ++i1) {        \
    KOKKOS_IMPL_NLWOTILE_L1(type, rp, func, i1, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_NLWOTILE_L3(type, rp, func, ...)         \
  for (auto i2 = static_cast<type>(rp.m_lower[2]);           \
       i2 < static_cast<type>(rp.m_upper[2]); ++i2) {        \
    KOKKOS_IMPL_NLWOTILE_L2(type, rp, func, i2, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_NLWOTILE_L4(type, rp, func, ...)         \
  for (auto i3 = static_cast<type>(rp.m_lower[3]);           \
       i3 < static_cast<type>(rp.m_upper[3]); ++i3) {        \
    KOKKOS_IMPL_NLWOTILE_L3(type, rp, func, i3, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_NLWOTILE_L5(type, rp, func, ...)         \
  for (auto i4 = static_cast<type>(rp.m_lower[4]);           \
       i4 < static_cast<type>(rp.m_upper[4]); ++i4) {        \
    KOKKOS_IMPL_NLWOTILE_L4(type, rp, func, i4, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_NLWOTILE_L6(type, rp, func, ...)         \
  for (auto i5 = static_cast<type>(rp.m_lower[5]);           \
       i5 < static_cast<type>(rp.m_upper[5]); ++i5) {        \
    KOKKOS_IMPL_NLWOTILE_L5(type, rp, func, i5, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_NLWOTILE_L7(type, rp, func, ...)         \
  for (auto i6 = static_cast<type>(rp.m_lower[6]);           \
       i6 < static_cast<type>(rp.m_upper[6]); ++i6) {        \
    KOKKOS_IMPL_NLWOTILE_L6(type, rp, func, i6, __VA_ARGS__) \
  }

// Non-Tagged: Outer Iteration: Left or Right

#define KOKKOS_IMPL_NLWOTILE_LOOP_1(type, is_left_outer, rp, func) \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                               \
  for (auto i0 = static_cast<type>(rp.m_lower[0]);                 \
       i0 < static_cast<type>(rp.m_upper[0]); ++i0) {              \
    KOKKOS_IMPL_NLWOTILE_APPLY(func, i0)                           \
  }

#define KOKKOS_IMPL_NLWOTILE_LOOP_2(type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                   \
    for (auto i1 = static_cast<type>(rp.m_lower[1]);               \
         i1 < static_cast<type>(rp.m_upper[1]); ++i1) {            \
      KOKKOS_IMPL_NLWOTILE_L1(type, rp, func, i1)                  \
    }                                                              \
  } else {                                                         \
    for (auto i1 = static_cast<type>(rp.m_lower[0]);               \
         i1 < static_cast<type>(rp.m_upper[0]); ++i1) {            \
      KOKKOS_IMPL_NLWOTILE_R1(type, rp, func, 2, i1)               \
    }                                                              \
  }

#define KOKKOS_IMPL_NLWOTILE_LOOP_3(type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                   \
    for (auto i2 = static_cast<type>(rp.m_lower[2]);               \
         i2 < static_cast<type>(rp.m_upper[2]); ++i2) {            \
      KOKKOS_IMPL_NLWOTILE_L2(type, rp, func, i2)                  \
    }                                                              \
  } else {                                                         \
    for (auto i2 = static_cast<type>(rp.m_lower[0]);               \
         i2 < static_cast<type>(rp.m_upper[0]); ++i2) {            \
      KOKKOS_IMPL_NLWOTILE_R2(type, rp, func, 3, i2)               \
    }                                                              \
  }

#define KOKKOS_IMPL_NLWOTILE_LOOP_4(type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                   \
    for (auto i3 = static_cast<type>(rp.m_lower[3]);               \
         i3 < static_cast<type>(rp.m_upper[3]); ++i3) {            \
      KOKKOS_IMPL_NLWOTILE_L3(type, rp, func, i3)                  \
    }                                                              \
  } else {                                                         \
    for (auto i3 = static_cast<type>(rp.m_lower[0]);               \
         i3 < static_cast<type>(rp.m_upper[0]); ++i3) {            \
      KOKKOS_IMPL_NLWOTILE_R3(type, rp, func, 4, i3)               \
    }                                                              \
  }

#define KOKKOS_IMPL_NLWOTILE_LOOP_5(type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                   \
    for (auto i4 = static_cast<type>(rp.m_lower[4]);               \
         i4 < static_cast<type>(rp.m_upper[4]); ++i4) {            \
      KOKKOS_IMPL_NLWOTILE_L4(type, rp, func, i4)                  \
    }                                                              \
  } else {                                                         \
    for (auto i4 = static_cast<type>(rp.m_lower[0]);               \
         i4 < static_cast<type>(rp.m_upper[0]); ++i4) {            \
      KOKKOS_IMPL_NLWOTILE_R4(type, rp, func, 5, i4)               \
    }                                                              \
  }

#define KOKKOS_IMPL_NLWOTILE_LOOP_6(type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                   \
    for (auto i5 = static_cast<type>(rp.m_lower[5]);               \
         i5 < static_cast<type>(rp.m_upper[5]); ++i5) {            \
      KOKKOS_IMPL_NLWOTILE_L5(type, rp, func, i5)                  \
    }                                                              \
  } else {                                                         \
    for (auto i5 = static_cast<type>(rp.m_lower[0]);               \
         i5 < static_cast<type>(rp.m_upper[0]); ++i5) {            \
      KOKKOS_IMPL_NLWOTILE_R5(type, rp, func, 6, i5)               \
    }                                                              \
  }

#define KOKKOS_IMPL_NLWOTILE_LOOP_7(type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                   \
    for (auto i6 = static_cast<type>(rp.m_lower[6]);               \
         i6 < static_cast<type>(rp.m_upper[6]); ++i6) {            \
      KOKKOS_IMPL_NLWOTILE_L6(type, rp, func, i6)                  \
    }                                                              \
  } else {                                                         \
    for (auto i6 = static_cast<type>(rp.m_lower[0]);               \
         i6 < static_cast<type>(rp.m_upper[0]); ++i6) {            \
      KOKKOS_IMPL_NLWOTILE_R6(type, rp, func, 7, i6)               \
    }                                                              \
  }

#define KOKKOS_IMPL_NLWOTILE_LOOP_8(type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                   \
    for (auto i7 = static_cast<type>(rp.m_lower[7]);               \
         i7 < static_cast<type>(rp.m_upper[7]); ++i7) {            \
      KOKKOS_IMPL_NLWOTILE_L7(type, rp, func, i7)                  \
    }                                                              \
  } else {                                                         \
    for (auto i7 = static_cast<type>(rp.m_lower[0]);               \
         i7 < static_cast<type>(rp.m_upper[0]); ++i7) {            \
      KOKKOS_IMPL_NLWOTILE_R7(type, rp, func, 8, i7)               \
    }                                                              \
  }

// ParallelFor macros: Tagged

#define KOKKOS_IMPL_TAGGED_NLWOTILE_APPLY(tag, func, ...) \
  func(tag, __VA_ARGS__);

// Tagged: Outer Iterate::Right
#define KOKKOS_IMPL_TAGGED_NLWOTILE_R1(tag, type, rp, func, rank, ...) \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                   \
  for (auto i0 = static_cast<type>(rp.m_lower[rank - 1]);              \
       i0 < static_cast<type>(rp.m_upper[rank - 1]); ++i0) {           \
    KOKKOS_IMPL_TAGGED_NLWOTILE_APPLY(tag, func, __VA_ARGS__, i0)      \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_R2(tag, type, rp, func, rank, ...)         \
  for (auto i1 = static_cast<type>(rp.m_lower[rank - 2]);                      \
       i1 < static_cast<type>(rp.m_upper[rank - 2]); ++i1) {                   \
    KOKKOS_IMPL_TAGGED_NLWOTILE_R1(tag, type, rp, func, rank, __VA_ARGS__, i1) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_R3(tag, type, rp, func, rank, ...)         \
  for (auto i2 = static_cast<type>(rp.m_lower[rank - 3]);                      \
       i2 < static_cast<type>(rp.m_upper[rank - 3]); ++i2) {                   \
    KOKKOS_IMPL_TAGGED_NLWOTILE_R2(tag, type, rp, func, rank, __VA_ARGS__, i2) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_R4(tag, type, rp, func, rank, ...)         \
  for (auto i3 = static_cast<type>(rp.m_lower[rank - 4]);                      \
       i3 < static_cast<type>(rp.m_upper[rank - 4]); ++i3) {                   \
    KOKKOS_IMPL_TAGGED_NLWOTILE_R3(tag, type, rp, func, rank, __VA_ARGS__, i3) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_R5(tag, type, rp, func, rank, ...)         \
  for (auto i4 = static_cast<type>(rp.m_lower[rank - 5]);                      \
       i4 < static_cast<type>(rp.m_upper[rank - 5]); ++i4) {                   \
    KOKKOS_IMPL_TAGGED_NLWOTILE_R4(tag, type, rp, func, rank, __VA_ARGS__, i4) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_R6(tag, type, rp, func, rank, ...)         \
  for (auto i5 = static_cast<type>(rp.m_lower[rank - 6]);                      \
       i5 < static_cast<type>(rp.m_upper[rank - 6]); ++i5) {                   \
    KOKKOS_IMPL_TAGGED_NLWOTILE_R5(tag, type, rp, func, rank, __VA_ARGS__, i5) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_R7(tag, type, rp, func, rank, ...)         \
  for (auto i6 = static_cast<type>(rp.m_lower[rank - 7]);                      \
       i6 < static_cast<type>(rp.m_upper[rank - 7]); ++i6) {                   \
    KOKKOS_IMPL_TAGGED_NLWOTILE_R6(tag, type, rp, func, rank, __VA_ARGS__, i6) \
  }

// Tagged: Outer Iterate::Left
#define KOKKOS_IMPL_TAGGED_NLWOTILE_L1(tag, type, rp, func, ...)  \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                              \
  for (auto i0 = static_cast<type>(rp.m_lower[0]);                \
       i0 < static_cast<type>(rp.m_upper[0]); ++i0) {             \
    KOKKOS_IMPL_TAGGED_NLWOTILE_APPLY(tag, func, i0, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_L2(tag, type, rp, func, ...)         \
  for (auto i1 = static_cast<type>(rp.m_lower[1]);                       \
       i1 < static_cast<type>(rp.m_upper[1]); ++i1) {                    \
    KOKKOS_IMPL_TAGGED_NLWOTILE_L1(tag, type, rp, func, i1, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_L3(tag, type, rp, func, ...)         \
  for (auto i2 = static_cast<type>(rp.m_lower[2]);                       \
       i2 < static_cast<type>(rp.m_upper[2]); ++i2) {                    \
    KOKKOS_IMPL_TAGGED_NLWOTILE_L2(tag, type, rp, func, i2, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_L4(tag, type, rp, func, ...)         \
  for (auto i3 = static_cast<type>(rp.m_lower[3]);                       \
       i3 < static_cast<type>(rp.m_upper[3]); ++i3) {                    \
    KOKKOS_IMPL_TAGGED_NLWOTILE_L3(tag, type, rp, func, i3, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_L5(tag, type, rp, func, ...)         \
  for (auto i4 = static_cast<type>(rp.m_lower[4]);                       \
       i4 < static_cast<type>(rp.m_upper[4]); ++i4) {                    \
    KOKKOS_IMPL_TAGGED_NLWOTILE_L4(tag, type, rp, func, i4, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_L6(tag, type, rp, func, ...)         \
  for (auto i5 = static_cast<type>(rp.m_lower[5]);                       \
       i5 < static_cast<type>(rp.m_upper[5]); ++i5) {                    \
    KOKKOS_IMPL_TAGGED_NLWOTILE_L5(tag, type, rp, func, i5, __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_L7(tag, type, rp, func, ...)         \
  for (auto i6 = static_cast<type>(rp.m_lower[6]);                       \
       i6 < static_cast<type>(rp.m_upper[6]); ++i6) {                    \
    KOKKOS_IMPL_TAGGED_NLWOTILE_L6(tag, type, rp, func, i6, __VA_ARGS__) \
  }

// Tagged: Outer Iterate: Left or Right

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_1(tag, type, is_left_outer, rp, func) \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                           \
  for (auto i0 = static_cast<type>(rp.m_lower[0]);                             \
       i0 < static_cast<type>(rp.m_upper[0]); ++i0) {                          \
    KOKKOS_IMPL_TAGGED_NLWOTILE_APPLY(tag, func, i0)                           \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_2(tag, type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                               \
    for (auto i1 = static_cast<type>(rp.m_lower[1]);                           \
         i1 < static_cast<type>(rp.m_upper[1]); ++i1) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_L1(tag, type, rp, func, i1)                  \
    }                                                                          \
  } else {                                                                     \
    for (auto i1 = static_cast<type>(rp.m_lower[0]);                           \
         i1 < static_cast<type>(rp.m_upper[0]); ++i1) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_R1(tag, type, rp, func, 2, i1)               \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_3(tag, type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                               \
    for (auto i2 = static_cast<type>(rp.m_lower[2]);                           \
         i2 < static_cast<type>(rp.m_upper[2]); ++i2) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_L2(tag, type, rp, func, i2)                  \
    }                                                                          \
  } else {                                                                     \
    for (auto i2 = static_cast<type>(rp.m_lower[0]);                           \
         i2 < static_cast<type>(rp.m_upper[0]); ++i2) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_R2(tag, type, rp, func, 3, i2)               \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_4(tag, type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                               \
    for (auto i3 = static_cast<type>(rp.m_lower[3]);                           \
         i3 < static_cast<type>(rp.m_upper[3]); ++i3) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_L3(tag, type, rp, func, i3)                  \
    }                                                                          \
  } else {                                                                     \
    for (auto i3 = static_cast<type>(rp.m_lower[0]);                           \
         i3 < static_cast<type>(rp.m_upper[0]); ++i3) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_R3(tag, type, rp, func, 4, i3)               \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_5(tag, type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                               \
    for (auto i4 = static_cast<type>(rp.m_lower[4]);                           \
         i4 < static_cast<type>(rp.m_upper[4]); ++i4) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_L4(tag, type, rp, func, i4)                  \
    }                                                                          \
  } else {                                                                     \
    for (auto i4 = static_cast<type>(rp.m_lower[0]);                           \
         i4 < static_cast<type>(rp.m_upper[0]); ++i4) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_R4(tag, type, rp, func, 5, i4)               \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_6(tag, type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                               \
    for (auto i5 = static_cast<type>(rp.m_lower[5]);                           \
         i5 < static_cast<type>(rp.m_upper[5]); ++i5) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_L5(tag, type, rp, func, i5)                  \
    }                                                                          \
  } else {                                                                     \
    for (auto i5 = static_cast<type>(rp.m_lower[0]);                           \
         i5 < static_cast<type>(rp.m_upper[0]); ++i5) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_R5(tag, type, rp, func, 6, i5)               \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_7(tag, type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                               \
    for (auto i6 = static_cast<type>(rp.m_lower[6]);                           \
         i6 < static_cast<type>(rp.m_upper[6]); ++i6) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_L6(tag, type, rp, func, i6)                  \
    }                                                                          \
  } else {                                                                     \
    for (auto i6 = static_cast<type>(rp.m_lower[0]);                           \
         i6 < static_cast<type>(rp.m_upper[0]); ++i6) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_R6(tag, type, rp, func, 7, i6)               \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_8(tag, type, is_left_outer, rp, func) \
  if constexpr (is_left_outer) {                                               \
    for (auto i7 = static_cast<type>(rp.m_lower[7]);                           \
         i7 < static_cast<type>(rp.m_upper[7]); ++i7) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_L7(tag, type, rp, func, i7)                  \
    }                                                                          \
  } else {                                                                     \
    for (auto i7 = static_cast<type>(rp.m_lower[0]);                           \
         i7 < static_cast<type>(rp.m_upper[0]); ++i7) {                        \
      KOKKOS_IMPL_TAGGED_NLWOTILE_R7(tag, type, rp, func, 8, i7)               \
    }                                                                          \
  }

// Rankwise structs for directly iterating over the elements via nested for
// loops
template <int Rank, bool IsLeftOuter, typename Tag, typename Enable = void>
struct Element_Loop_Type;

template <bool IsLeftOuter>
struct Element_Loop_Type<1, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_1(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter>
struct Element_Loop_Type<2, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_2(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter>
struct Element_Loop_Type<3, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_3(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter>
struct Element_Loop_Type<4, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_4(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter>
struct Element_Loop_Type<5, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_5(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter>
struct Element_Loop_Type<6, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_6(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter>
struct Element_Loop_Type<7, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_7(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter>
struct Element_Loop_Type<8, IsLeftOuter, void, void> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_NLWOTILE_LOOP_8(typename RP::index_type, IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<1, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_1(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<2, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_2(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<3, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_3(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<4, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_4(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<5, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_5(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<6, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_6(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<7, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_7(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

template <bool IsLeftOuter, typename Tag>
struct Element_Loop_Type<8, IsLeftOuter, Tag,
                         std::enable_if_t<!std::is_void_v<Tag>>> {
  template <typename RP, typename Functor>
  static void apply(RP const& rp, Functor const& func) {
    KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_8(Tag{}, typename RP::index_type,
                                       IsLeftOuter, rp, func);
  }
};

// Primary template for iterating via a nested loop without tiles
template <typename RP, typename Functor, typename Tag = void,
          typename ValueType = void, typename Enable = void>
struct HostIterateNestLoopWoTile;

// Specialization for ParallelFor
template <typename RP, typename Functor, typename Tag, typename ValueType>
struct HostIterateNestLoopWoTile<RP, Functor, Tag, ValueType,
                                 std::enable_if_t<std::is_void_v<ValueType>>> {
  inline HostIterateNestLoopWoTile(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  inline void execute() const {
    if (m_rp.m_num_tiles > 0) {
      Element_Loop_Type<RP::rank, (RP::outer_direction == Iterate::Left),
                        Tag>::apply(m_rp, m_func);
    }
  }

  const RP m_rp;
  const Functor m_func;
};

// TODO: HostIterateNestLoopWoTile specializations for Parallel Reduce.

// ------------------------------------------------------------------ //

#undef KOKKOS_IMPL_NLWOTILE_APPLY
#undef KOKKOS_IMPL_NLWOTILE_L1
#undef KOKKOS_IMPL_NLWOTILE_L2
#undef KOKKOS_IMPL_NLWOTILE_L3
#undef KOKKOS_IMPL_NLWOTILE_L4
#undef KOKKOS_IMPL_NLWOTILE_L5
#undef KOKKOS_IMPL_NLWOTILE_L6
#undef KOKKOS_IMPL_NLWOTILE_L7
#undef KOKKOS_IMPL_NLWOTILE_L8
#undef KOKKOS_IMPL_NLWOTILE_R1
#undef KOKKOS_IMPL_NLWOTILE_R2
#undef KOKKOS_IMPL_NLWOTILE_R3
#undef KOKKOS_IMPL_NLWOTILE_R4
#undef KOKKOS_IMPL_NLWOTILE_R5
#undef KOKKOS_IMPL_NLWOTILE_R6
#undef KOKKOS_IMPL_NLWOTILE_R7
#undef KOKKOS_IMPL_NLWOTILE_R8
#undef KOKKOS_IMPL_NLWOTILE_LOOP_1
#undef KOKKOS_IMPL_NLWOTILE_LOOP_2
#undef KOKKOS_IMPL_NLWOTILE_LOOP_3
#undef KOKKOS_IMPL_NLWOTILE_LOOP_4
#undef KOKKOS_IMPL_NLWOTILE_LOOP_5
#undef KOKKOS_IMPL_NLWOTILE_LOOP_6
#undef KOKKOS_IMPL_NLWOTILE_LOOP_7
#undef KOKKOS_IMPL_NLWOTILE_LOOP_8
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_APPLY
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L1
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L2
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L3
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L4
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L5
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L6
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L7
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_L8
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R1
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R2
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R3
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R4
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R5
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R6
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R7
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_R8
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_1
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_2
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_3
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_4
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_5
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_6
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_7
#undef KOKKOS_IMPL_TAGGED_NLWOTILE_LOOP_8

}  // namespace Impl
}  // namespace Kokkos

#endif
