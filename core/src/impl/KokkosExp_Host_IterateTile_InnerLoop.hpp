// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HOST_EXP_ITERATE_TILE_INNER_LOOP_HPP
#define KOKKOS_HOST_EXP_ITERATE_TILE_INNER_LOOP_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP) && !defined(__CUDA_ARCH__)
#define KOKKOS_MDRANGE_IVDEP
#endif

#ifdef KOKKOS_MDRANGE_IVDEP
#define KOKKOS_ENABLE_IVDEP_MDRANGE _Pragma("ivdep")
#else
#define KOKKOS_ENABLE_IVDEP_MDRANGE
#endif

#include <algorithm>

namespace Kokkos {
namespace Impl {

// New Loop Macros...
// parallel_for, non-tagged
#define KOKKOS_IMPL_APPLY(func, ...) func(__VA_ARGS__);

// LayoutRight
// d = 0 to start

#define KOKKOS_IMPL_LOOP_R_2(func, type, m_offset, extent, d, ...)         \
  for (type i1 = (type)0; i1 < static_cast<type>(extent[d]); ++i1) {       \
    KOKKOS_IMPL_APPLY(func, __VA_ARGS__, i1 + m_offset[d])                 \
  }

#define KOKKOS_IMPL_LOOP_R_3(func, type, m_offset, extent, d, ...)         \
  for (type i2 = (type)0; i2 < static_cast<type>(extent[d]); ++i2) {       \
    KOKKOS_IMPL_LOOP_R_2(func, type, m_offset, extent, d + 1, __VA_ARGS__, \
                         i2 + m_offset[d])                                 \
  }

#define KOKKOS_IMPL_LOOP_R_4(func, type, m_offset, extent, d, ...)         \
  for (type i3 = (type)0; i3 < static_cast<type>(extent[d]); ++i3) {       \
    KOKKOS_IMPL_LOOP_R_3(func, type, m_offset, extent, d + 1, __VA_ARGS__, \
                         i3 + m_offset[d])                                 \
  }

#define KOKKOS_IMPL_LOOP_R_5(func, type, m_offset, extent, d, ...)         \
  for (type i4 = (type)0; i4 < static_cast<type>(extent[d]); ++i4) {       \
    KOKKOS_IMPL_LOOP_R_4(func, type, m_offset, extent, d + 1, __VA_ARGS__, \
                         i4 + m_offset[d])                                 \
  }

#define KOKKOS_IMPL_LOOP_R_6(func, type, m_offset, extent, d, ...)         \
  for (type i5 = (type)0; i5 < static_cast<type>(extent[d]); ++i5) {       \
    KOKKOS_IMPL_LOOP_R_5(func, type, m_offset, extent, d + 1, __VA_ARGS__, \
                         i5 + m_offset[d])                                 \
  }

#define KOKKOS_IMPL_LOOP_R_7(func, type, m_offset, extent, d, ...)         \
  for (type i6 = (type)0; i6 < static_cast<type>(extent[d]); ++i6) {       \
    KOKKOS_IMPL_LOOP_R_6(func, type, m_offset, extent, d + 1, __VA_ARGS__, \
                         i6 + m_offset[d])                                 \
  }

#define KOKKOS_IMPL_LOOP_R_8(func, type, m_offset, extent, d, ...)         \
  for (type i7 = (type)0; i7 < static_cast<type>(extent[d]); ++i7) {       \
    KOKKOS_IMPL_LOOP_R_7(func, type, m_offset, extent, d + 1, __VA_ARGS__, \
                         i7 + m_offset[d])                                 \
  }

// LayoutLeft
// d = rank-1 to start

#define KOKKOS_IMPL_LOOP_L_2(func, type, m_offset, extent, d, ...)   \
  for (type i1 = (type)0; i1 < static_cast<type>(extent[d]); ++i1) { \
    KOKKOS_IMPL_APPLY(func, i1 + m_offset[d], __VA_ARGS__)           \
  }

#define KOKKOS_IMPL_LOOP_L_3(func, type, m_offset, extent, d, ...)   \
  for (type i2 = (type)0; i2 < static_cast<type>(extent[d]); ++i2) { \
    KOKKOS_IMPL_LOOP_L_2(func, type, m_offset, extent, d - 1,        \
                         i2 + m_offset[d], __VA_ARGS__)              \
  }

#define KOKKOS_IMPL_LOOP_L_4(func, type, m_offset, extent, d, ...)   \
  for (type i3 = (type)0; i3 < static_cast<type>(extent[d]); ++i3) { \
    KOKKOS_IMPL_LOOP_L_3(func, type, m_offset, extent, d - 1,        \
                         i3 + m_offset[d], __VA_ARGS__)              \
  }

#define KOKKOS_IMPL_LOOP_L_5(func, type, m_offset, extent, d, ...)   \
  for (type i4 = (type)0; i4 < static_cast<type>(extent[d]); ++i4) { \
    KOKKOS_IMPL_LOOP_L_4(func, type, m_offset, extent, d - 1,        \
                         i4 + m_offset[d], __VA_ARGS__)              \
  }

#define KOKKOS_IMPL_LOOP_L_6(func, type, m_offset, extent, d, ...)   \
  for (type i5 = (type)0; i5 < static_cast<type>(extent[d]); ++i5) { \
    KOKKOS_IMPL_LOOP_L_5(func, type, m_offset, extent, d - 1,        \
                         i5 + m_offset[d], __VA_ARGS__)              \
  }

#define KOKKOS_IMPL_LOOP_L_7(func, type, m_offset, extent, d, ...)   \
  for (type i6 = (type)0; i6 < static_cast<type>(extent[d]); ++i6) { \
    KOKKOS_IMPL_LOOP_L_6(func, type, m_offset, extent, d - 1,        \
                         i6 + m_offset[d], __VA_ARGS__)              \
  }

#define KOKKOS_IMPL_LOOP_L_8(func, type, m_offset, extent, d, ...)   \
  for (type i7 = (type)0; i7 < static_cast<type>(extent[d]); ++i7) { \
    KOKKOS_IMPL_LOOP_L_7(func, type, m_offset, extent, d - 1,        \
                         i7 + m_offset[d], __VA_ARGS__)              \
  }

// Left vs Right
// TODO: rank not necessary to pass through, can hardcode the values
#define KOKKOS_IMPL_LOOP_LAYOUT_1(func, type, is_left, m_offset, extent, rank) \
    KOKKOS_IMPL_APPLY(func)

#define KOKKOS_IMPL_LOOP_LAYOUT_2(func, type, is_left, m_offset, extent, rank) \
  if (is_left) {                                                               \
    for (type i1 = (type)0; i1 < static_cast<type>(extent[rank - 1]); ++i1) {  \
      KOKKOS_IMPL_APPLY(func, i1 + m_offset[rank - 1])                         \
    }                                                                          \
  } else {                                                                     \
    for (type i1 = (type)0; i1 < static_cast<type>(extent[0]); ++i1) {         \
      KOKKOS_IMPL_APPLY(func, i1 + m_offset[0])                                \
    }                                                                          \
  }

#define KOKKOS_IMPL_LOOP_LAYOUT_3(func, type, is_left, m_offset, extent, rank) \
  if (is_left) {                                                               \
    for (type i2 = (type)0; i2 < static_cast<type>(extent[rank - 1]); ++i2) {  \
      KOKKOS_IMPL_LOOP_L_2(func, type, m_offset, extent, rank - 2,             \
                           i2 + m_offset[rank - 1])                            \
    }                                                                          \
  } else {                                                                     \
    for (type i2 = (type)0; i2 < static_cast<type>(extent[0]); ++i2) {         \
      KOKKOS_IMPL_LOOP_R_2(func, type, m_offset, extent, 1, i2 + m_offset[0])  \
    }                                                                          \
  }

#define KOKKOS_IMPL_LOOP_LAYOUT_4(func, type, is_left, m_offset, extent, rank) \
  if (is_left) {                                                               \
    for (type i3 = (type)0; i3 < static_cast<type>(extent[rank - 1]); ++i3) {  \
      KOKKOS_IMPL_LOOP_L_3(func, type, m_offset, extent, rank - 2,             \
                           i3 + m_offset[rank - 1])                            \
    }                                                                          \
  } else {                                                                     \
    for (type i3 = (type)0; i3 < static_cast<type>(extent[0]); ++i3) {         \
      KOKKOS_IMPL_LOOP_R_3(func, type, m_offset, extent, 1, i3 + m_offset[0])  \
    }                                                                          \
  }

#define KOKKOS_IMPL_LOOP_LAYOUT_5(func, type, is_left, m_offset, extent, rank) \
  if (is_left) {                                                               \
    for (type i4 = (type)0; i4 < static_cast<type>(extent[rank - 1]); ++i4) {  \
      KOKKOS_IMPL_LOOP_L_4(func, type, m_offset, extent, rank - 2,             \
                           i4 + m_offset[rank - 1])                            \
    }                                                                          \
  } else {                                                                     \
    for (type i4 = (type)0; i4 < static_cast<type>(extent[0]); ++i4) {         \
      KOKKOS_IMPL_LOOP_R_4(func, type, m_offset, extent, 1, i4 + m_offset[0])  \
    }                                                                          \
  }

#define KOKKOS_IMPL_LOOP_LAYOUT_6(func, type, is_left, m_offset, extent, rank) \
  if (is_left) {                                                               \
    for (type i5 = (type)0; i5 < static_cast<type>(extent[rank - 1]); ++i5) {  \
      KOKKOS_IMPL_LOOP_L_5(func, type, m_offset, extent, rank - 2,             \
                           i5 + m_offset[rank - 1])                            \
    }                                                                          \
  } else {                                                                     \
    for (type i5 = (type)0; i5 < static_cast<type>(extent[0]); ++i5) {         \
      KOKKOS_IMPL_LOOP_R_5(func, type, m_offset, extent, 1, i5 + m_offset[0])  \
    }                                                                          \
  }

#define KOKKOS_IMPL_LOOP_LAYOUT_7(func, type, is_left, m_offset, extent, rank) \
  if (is_left) {                                                               \
    for (type i6 = (type)0; i6 < static_cast<type>(extent[rank - 1]); ++i6) {  \
      KOKKOS_IMPL_LOOP_L_6(func, type, m_offset, extent, rank - 2,             \
                           i6 + m_offset[rank - 1])                            \
    }                                                                          \
  } else {                                                                     \
    for (type i6 = (type)0; i6 < static_cast<type>(extent[0]); ++i6) {         \
      KOKKOS_IMPL_LOOP_R_6(func, type, m_offset, extent, 1, i6 + m_offset[0])  \
    }                                                                          \
  }

#define KOKKOS_IMPL_LOOP_LAYOUT_8(func, type, is_left, m_offset, extent, rank) \
  if (is_left) {                                                               \
    for (type i7 = (type)0; i7 < static_cast<type>(extent[rank - 1]); ++i7) {  \
      KOKKOS_IMPL_LOOP_L_7(func, type, m_offset, extent, rank - 2,             \
                           i7 + m_offset[rank - 1])                            \
    }                                                                          \
  } else {                                                                     \
    for (type i7 = (type)0; i7 < static_cast<type>(extent[0]); ++i7) {         \
      KOKKOS_IMPL_LOOP_R_7(func, type, m_offset, extent, 1, i7 + m_offset[0])  \
    }                                                                          \
  }

// Partial vs Full Tile
// hack
#define KOKKOS_IMPL_TILE_LOOP_1(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  func();

#define KOKKOS_IMPL_TILE_LOOP_2(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  if (cond) {                                                                \
    KOKKOS_IMPL_LOOP_LAYOUT_2(func, type, is_left, m_offset, extent_full,    \
                              rank)                                          \
  } else {                                                                   \
    KOKKOS_IMPL_LOOP_LAYOUT_2(func, type, is_left, m_offset, extent_partial, \
                              rank)                                          \
  }

#define KOKKOS_IMPL_TILE_LOOP_3(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  if (cond) {                                                                \
    KOKKOS_IMPL_LOOP_LAYOUT_3(func, type, is_left, m_offset, extent_full,    \
                              rank)                                          \
  } else {                                                                   \
    KOKKOS_IMPL_LOOP_LAYOUT_3(func, type, is_left, m_offset, extent_partial, \
                              rank)                                          \
  }

#define KOKKOS_IMPL_TILE_LOOP_4(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  if (cond) {                                                                \
    KOKKOS_IMPL_LOOP_LAYOUT_4(func, type, is_left, m_offset, extent_full,    \
                              rank)                                          \
  } else {                                                                   \
    KOKKOS_IMPL_LOOP_LAYOUT_4(func, type, is_left, m_offset, extent_partial, \
                              rank)                                          \
  }

#define KOKKOS_IMPL_TILE_LOOP_5(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  if (cond) {                                                                \
    KOKKOS_IMPL_LOOP_LAYOUT_5(func, type, is_left, m_offset, extent_full,    \
                              rank)                                          \
  } else {                                                                   \
    KOKKOS_IMPL_LOOP_LAYOUT_5(func, type, is_left, m_offset, extent_partial, \
                              rank)                                          \
  }

#define KOKKOS_IMPL_TILE_LOOP_6(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  if (cond) {                                                                \
    KOKKOS_IMPL_LOOP_LAYOUT_6(func, type, is_left, m_offset, extent_full,    \
                              rank)                                          \
  } else {                                                                   \
    KOKKOS_IMPL_LOOP_LAYOUT_6(func, type, is_left, m_offset, extent_partial, \
                              rank)                                          \
  }

#define KOKKOS_IMPL_TILE_LOOP_7(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  if (cond) {                                                                \
    KOKKOS_IMPL_LOOP_LAYOUT_7(func, type, is_left, m_offset, extent_full,    \
                              rank)                                          \
  } else {                                                                   \
    KOKKOS_IMPL_LOOP_LAYOUT_7(func, type, is_left, m_offset, extent_partial, \
                              rank)                                          \
  }

#define KOKKOS_IMPL_TILE_LOOP_8(func, type, is_left, cond, m_offset,         \
                                extent_full, extent_partial, rank)           \
  if (cond) {                                                                \
    KOKKOS_IMPL_LOOP_LAYOUT_8(func, type, is_left, m_offset, extent_full,    \
                              rank)                                          \
  } else {                                                                   \
    KOKKOS_IMPL_LOOP_LAYOUT_8(func, type, is_left, m_offset, extent_partial, \
                              rank)                                          \
  }

// tagged macros
#define KOKKOS_IMPL_TAGGED_APPLY(tag, func, ...) func(tag, __VA_ARGS__);

// LayoutRight
// d = 0 to start

#define KOKKOS_IMPL_TAGGED_LOOP_R_2(tag, func, type, m_offset, extent, d, ...) \
  for (type i1 = (type)0; i1 < static_cast<type>(extent[d]); ++i1) {           \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, __VA_ARGS__, i1 + m_offset[d])         \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_R_3(tag, func, type, m_offset, extent, d, ...) \
  for (type i2 = (type)0; i2 < static_cast<type>(extent[d]); ++i2) {           \
    KOKKOS_IMPL_TAGGED_LOOP_R_2(tag, func, type, m_offset, extent, d + 1,      \
                                __VA_ARGS__, i2 + m_offset[d])                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_R_4(tag, func, type, m_offset, extent, d, ...) \
  for (type i3 = (type)0; i3 < static_cast<type>(extent[d]); ++i3) {           \
    KOKKOS_IMPL_TAGGED_LOOP_R_3(tag, func, type, m_offset, extent, d + 1,      \
                                __VA_ARGS__, i3 + m_offset[d])                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_R_5(tag, func, type, m_offset, extent, d, ...) \
  for (type i4 = (type)0; i4 < static_cast<type>(extent[d]); ++i4) {           \
    KOKKOS_IMPL_TAGGED_LOOP_R_4(tag, func, type, m_offset, extent, d + 1,      \
                                __VA_ARGS__, i4 + m_offset[d])                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_R_6(tag, func, type, m_offset, extent, d, ...) \
  for (type i5 = (type)0; i5 < static_cast<type>(extent[d]); ++i5) {           \
    KOKKOS_IMPL_TAGGED_LOOP_R_5(tag, func, type, m_offset, extent, d + 1,      \
                                __VA_ARGS__, i5 + m_offset[d])                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_R_7(tag, func, type, m_offset, extent, d, ...) \
  for (type i6 = (type)0; i6 < static_cast<type>(extent[d]); ++i6) {           \
    KOKKOS_IMPL_TAGGED_LOOP_R_6(tag, func, type, m_offset, extent, d + 1,      \
                                __VA_ARGS__, i6 + m_offset[d])                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_R_8(tag, func, type, m_offset, extent, d, ...) \
  for (type i7 = (type)0; i7 < static_cast<type>(extent[d]); ++i7) {           \
    KOKKOS_IMPL_TAGGED_LOOP_R_7(tag, func, type, m_offset, extent, d + 1,      \
                                __VA_ARGS__, i7 + m_offset[d])                 \
  }

// LayoutLeft
// d = rank-1 to start
#define KOKKOS_IMPL_TAGGED_LOOP_L_2(tag, func, type, m_offset, extent, d, ...) \
  for (type i1 = (type)0; i1 < static_cast<type>(extent[d]); ++i1) {           \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, i1 + m_offset[d], __VA_ARGS__)         \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_L_3(tag, func, type, m_offset, extent, d, ...) \
  for (type i2 = (type)0; i2 < static_cast<type>(extent[d]); ++i2) {           \
    KOKKOS_IMPL_TAGGED_LOOP_L_2(tag, func, type, m_offset, extent, d - 1,      \
                                i2 + m_offset[d], __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_L_4(tag, func, type, m_offset, extent, d, ...) \
  for (type i3 = (type)0; i3 < static_cast<type>(extent[d]); ++i3) {           \
    KOKKOS_IMPL_TAGGED_LOOP_L_3(tag, func, type, m_offset, extent, d - 1,      \
                                i3 + m_offset[d], __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_L_5(tag, func, type, m_offset, extent, d, ...) \
  for (type i4 = (type)0; i4 < static_cast<type>(extent[d]); ++i4) {           \
    KOKKOS_IMPL_TAGGED_LOOP_L_4(tag, func, type, m_offset, extent, d - 1,      \
                                i4 + m_offset[d], __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_L_6(tag, func, type, m_offset, extent, d, ...) \
  for (type i5 = (type)0; i5 < static_cast<type>(extent[d]); ++i5) {           \
    KOKKOS_IMPL_TAGGED_LOOP_L_5(tag, func, type, m_offset, extent, d - 1,      \
                                i5 + m_offset[d], __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_L_7(tag, func, type, m_offset, extent, d, ...) \
  for (type i6 = (type)0; i6 < static_cast<type>(extent[d]); ++i6) {           \
    KOKKOS_IMPL_TAGGED_LOOP_L_6(tag, func, type, m_offset, extent, d - 1,      \
                                i6 + m_offset[d], __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_L_8(tag, func, type, m_offset, extent, d, ...) \
  for (type i7 = (type)0; i7 < static_cast<type>(extent[d]); ++i7) {           \
    KOKKOS_IMPL_TAGGED_LOOP_L_7(tag, func, type, m_offset, extent, d - 1,      \
                                i7 + m_offset[d], __VA_ARGS__)                 \
  }

// Left vs Right
// TODO: rank not necessary to pass through, can hardcode the values

#define KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_2(tag, func, type, is_left, m_offset,   \
                                         extent, rank)                         \
  if (is_left) {                                                               \
    for (type i1 = (type)0; i1 < static_cast<type>(extent[rank - 1]); ++i1) {  \
      KOKKOS_IMPL_TAGGED_APPLY(tag, func, i1 + m_offset[rank - 1])             \
    }                                                                          \
  } else {                                                                     \
    for (type i1 = (type)0; i1 < static_cast<type>(extent[0]); ++i1) {         \
      KOKKOS_IMPL_TAGGED_APPLY(tag, func, i1 + m_offset[0])                    \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_3(tag, func, type, is_left, m_offset,   \
                                         extent, rank)                         \
  if (is_left) {                                                               \
    for (type i2 = (type)0; i2 < static_cast<type>(extent[rank - 1]); ++i2) {  \
      KOKKOS_IMPL_TAGGED_LOOP_L_2(tag, func, type, m_offset, extent, rank - 2, \
                                  i2 + m_offset[rank - 1])                     \
    }                                                                          \
  } else {                                                                     \
    for (type i2 = (type)0; i2 < static_cast<type>(extent[0]); ++i2) {         \
      KOKKOS_IMPL_TAGGED_LOOP_R_2(tag, func, type, m_offset, extent, 1,        \
                                  i2 + m_offset[0])                            \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_4(tag, func, type, is_left, m_offset,   \
                                         extent, rank)                         \
  if (is_left) {                                                               \
    for (type i3 = (type)0; i3 < static_cast<type>(extent[rank - 1]); ++i3) {  \
      KOKKOS_IMPL_TAGGED_LOOP_L_3(tag, func, type, m_offset, extent, rank - 2, \
                                  i3 + m_offset[rank - 1])                     \
    }                                                                          \
  } else {                                                                     \
    for (type i3 = (type)0; i3 < static_cast<type>(extent[0]); ++i3) {         \
      KOKKOS_IMPL_TAGGED_LOOP_R_3(tag, func, type, m_offset, extent, 1,        \
                                  i3 + m_offset[0])                            \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_5(tag, func, type, is_left, m_offset,   \
                                         extent, rank)                         \
  if (is_left) {                                                               \
    for (type i4 = (type)0; i4 < static_cast<type>(extent[rank - 1]); ++i4) {  \
      KOKKOS_IMPL_TAGGED_LOOP_L_4(tag, func, type, m_offset, extent, rank - 2, \
                                  i4 + m_offset[rank - 1])                     \
    }                                                                          \
  } else {                                                                     \
    for (type i4 = (type)0; i4 < static_cast<type>(extent[0]); ++i4) {         \
      KOKKOS_IMPL_TAGGED_LOOP_R_4(tag, func, type, m_offset, extent, 1,        \
                                  i4 + m_offset[0])                            \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_6(tag, func, type, is_left, m_offset,   \
                                         extent, rank)                         \
  if (is_left) {                                                               \
    for (type i5 = (type)0; i5 < static_cast<type>(extent[rank - 1]); ++i5) {  \
      KOKKOS_IMPL_TAGGED_LOOP_L_5(tag, func, type, m_offset, extent, rank - 2, \
                                  i5 + m_offset[rank - 1])                     \
    }                                                                          \
  } else {                                                                     \
    for (type i5 = (type)0; i5 < static_cast<type>(extent[0]); ++i5) {         \
      KOKKOS_IMPL_TAGGED_LOOP_R_5(tag, func, type, m_offset, extent, 1,        \
                                  i5 + m_offset[0])                            \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_7(tag, func, type, is_left, m_offset,   \
                                         extent, rank)                         \
  if (is_left) {                                                               \
    for (type i6 = (type)0; i6 < static_cast<type>(extent[rank - 1]); ++i6) {  \
      KOKKOS_IMPL_TAGGED_LOOP_L_6(tag, func, type, m_offset, extent, rank - 2, \
                                  i6 + m_offset[rank - 1])                     \
    }                                                                          \
  } else {                                                                     \
    for (type i6 = (type)0; i6 < static_cast<type>(extent[0]); ++i6) {         \
      KOKKOS_IMPL_TAGGED_LOOP_R_6(tag, func, type, m_offset, extent, 1,        \
                                  i6 + m_offset[0])                            \
    }                                                                          \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_8(tag, func, type, is_left, m_offset,   \
                                         extent, rank)                         \
  if (is_left) {                                                               \
    for (type i7 = (type)0; i7 < static_cast<type>(extent[rank - 1]); ++i7) {  \
      KOKKOS_IMPL_TAGGED_LOOP_L_7(tag, func, type, m_offset, extent, rank - 2, \
                                  i7 + m_offset[rank - 1])                     \
    }                                                                          \
  } else {                                                                     \
    for (type i7 = (type)0; i7 < static_cast<type>(extent[0]); ++i7) {         \
      KOKKOS_IMPL_TAGGED_LOOP_R_7(tag, func, type, m_offset, extent, 1,        \
                                  i7 + m_offset[0])                            \
    }                                                                          \
  }

// Partial vs Full Tile
// hack
#define KOKKOS_IMPL_TAGGED_TILE_LOOP_1(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  func(tag);

#define KOKKOS_IMPL_TAGGED_TILE_LOOP_2(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  if (cond) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_2(tag, func, type, is_left, m_offset,      \
                                     extent_full, rank)                       \
  } else {                                                                    \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_2(tag, func, type, is_left, m_offset,      \
                                     extent_partial, rank)                    \
  }

#define KOKKOS_IMPL_TAGGED_TILE_LOOP_3(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  if (cond) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_3(tag, func, type, is_left, m_offset,      \
                                     extent_full, rank)                       \
  } else {                                                                    \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_3(tag, func, type, is_left, m_offset,      \
                                     extent_partial, rank)                    \
  }

#define KOKKOS_IMPL_TAGGED_TILE_LOOP_4(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  if (cond) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_4(tag, func, type, is_left, m_offset,      \
                                     extent_full, rank)                       \
  } else {                                                                    \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_4(tag, func, type, is_left, m_offset,      \
                                     extent_partial, rank)                    \
  }

#define KOKKOS_IMPL_TAGGED_TILE_LOOP_5(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  if (cond) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_5(tag, func, type, is_left, m_offset,      \
                                     extent_full, rank)                       \
  } else {                                                                    \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_5(tag, func, type, is_left, m_offset,      \
                                     extent_partial, rank)                    \
  }

#define KOKKOS_IMPL_TAGGED_TILE_LOOP_6(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  if (cond) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_6(tag, func, type, is_left, m_offset,      \
                                     extent_full, rank)                       \
  } else {                                                                    \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_6(tag, func, type, is_left, m_offset,      \
                                     extent_partial, rank)                    \
  }

#define KOKKOS_IMPL_TAGGED_TILE_LOOP_7(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  if (cond) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_7(tag, func, type, is_left, m_offset,      \
                                     extent_full, rank)                       \
  } else {                                                                    \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_7(tag, func, type, is_left, m_offset,      \
                                     extent_partial, rank)                    \
  }

#define KOKKOS_IMPL_TAGGED_TILE_LOOP_8(tag, func, type, is_left, cond,        \
                                       m_offset, extent_full, extent_partial, \
                                       rank)                                  \
  if (cond) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_8(tag, func, type, is_left, m_offset,      \
                                     extent_full, rank)                       \
  } else {                                                                    \
    KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_8(tag, func, type, is_left, m_offset,      \
                                     extent_partial, rank)                    \
  }

// end tagged macros

// Structs for calling loops
template <int Rank, bool IsLeft, typename IType, typename Tagged,
          typename Enable = void>
struct Tile_Inner_Loop_Type;

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<1, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_1(func, IType, IsLeft, cond, offset, a, b, 1);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<2, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_2(func, IType, IsLeft, cond, offset, a, b, 2);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<3, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_3(func, IType, IsLeft, cond, offset, a, b, 3);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<4, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_4(func, IType, IsLeft, cond, offset, a, b, 4);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<5, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_5(func, IType, IsLeft, cond, offset, a, b, 5);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<6, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_6(func, IType, IsLeft, cond, offset, a, b, 6);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<7, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_7(func, IType, IsLeft, cond, offset, a, b, 7);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Inner_Loop_Type<8, IsLeft, IType, void, void> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TILE_LOOP_8(func, IType, IsLeft, cond, offset, a, b, 8);
  }
};

// tagged versions

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<1, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_1(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 1);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<2, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_2(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 2);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<3, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_3(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 3);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<4, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_4(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 4);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<5, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_5(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 5);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<6, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_6(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 6);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<7, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_7(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 7);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Inner_Loop_Type<8, IsLeft, IType, Tagged,
                      std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename Offset, typename ExtentA, typename ExtentB>
  static void apply(Func const& func, bool cond, Offset const& offset,
                    ExtentA const& a, ExtentB const& b) {
    KOKKOS_IMPL_TAGGED_TILE_LOOP_8(Tagged(), func, IType, IsLeft, cond, offset,
                                   a, b, 8);
  }
};
// end Structs for calling loops

template <typename RP, typename Functor, typename Tag = void,
          typename ValueType = void, typename Enable = void>
struct HostIterTileLoopFunctor;

// For ParallelFor
template <typename RP, typename Functor, typename Tag, typename ValueType>
struct HostIterTileLoopFunctor<RP, Functor, Tag, ValueType,
                       std::enable_if_t<std::is_void_v<ValueType>>> {
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  inline HostIterTileLoopFunctor(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {

    if constexpr (RP::outer_direction == Iterate::Left) {
      m_num_outer_iters = m_rp.m_num_tiles / m_rp.m_tile_end[0];
    } else {
      m_num_outer_iters = m_rp.m_num_tiles / m_rp.m_tile_end[RP::rank - 1];
    }
  }

  inline bool check_iteration_bounds(point_type& partial_tile,
                                     const point_type& offset) const {
    bool is_full_tile = true;

    for (int i = 0; i < RP::rank; ++i) {
      if ((offset[i] + m_rp.m_tile[i]) <= m_rp.m_upper[i]) {
        partial_tile[i] = m_rp.m_tile[i];
      } else {
        is_full_tile = false;
        partial_tile[i] =
            m_rp.m_upper[i] - offset[i];  // remaining elements in dimension i
      }
    }

    return is_full_tile;
  }  // end check bounds

  template <int Rank>
  struct RankTag {
    using type = RankTag<Rank>;
    enum { value = (int)Rank };
  };

  template <typename... Args>
  std::enable_if_t<(sizeof...(Args) == (RP::rank-1) && std::is_void_v<Tag>), void>
  m_func_innermost_loop(Args... args) {
    if constexpr (RP::outer_direction == Iterate::Left) {
      // hack: it should be limited to a tile, instead
      // of the whole dimension 
      KOKKOS_ENABLE_IVDEP_MDRANGE
      for (int i = m_rp.m_lower[0]; i < m_rp.m_upper[0]; ++i) { 
        m_func(i, args...); 
      }
    } else {
      // hack: it should be limited to a tile, instead
      // of the whole dimension 
      KOKKOS_ENABLE_IVDEP_MDRANGE
      for (int i = m_rp.m_lower[RP::rank - 1]; i < m_rp.m_upper[RP::rank - 1]; ++i) { 
        m_func(args..., i); 
      }
    }
  }

  template <typename... Args>
  std::enable_if_t<(sizeof...(Args) == (RP::rank-1) && !std::is_void_v<Tag>), void>
  m_func_innermost_loop(Tag tag, Args... args) {
    if constexpr (RP::outer_direction == Iterate::Left) {
      // hack: it should be limited to a tile, instead
      // of the whole dimension 
      KOKKOS_ENABLE_IVDEP_MDRANGE
      for (int i = m_rp.m_lower[0]; i < m_rp.m_upper[0]; ++i) { 
        m_func(tag, i, args...); 
      }
    } else {
      // hack: it should be limited to a tile, instead
      // of the whole dimension 
      KOKKOS_ENABLE_IVDEP_MDRANGE
      for (int i = m_rp.m_lower[RP::rank - 1]; i < m_rp.m_upper[RP::rank - 1]; ++i) { 
        m_func(tag, args..., i); 
      }
    }
  }

  template <typename IType>
  inline void operator()(IType tile_idx) const {
    point_type m_offset;
    point_type m_tiledims;

    if constexpr (RP::outer_direction == Iterate::Left) {
      tile_idx *= m_rp.m_tile_end[0]; // hack
      for (int i = 0; i < RP::rank; ++i) {
        m_offset[i] =
            (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i];
        tile_idx /= m_rp.m_tile_end[i];
      }
    } else {
      tile_idx *= m_rp.m_tile_end[RP::rank - 1]; // hack
      for (int i = RP::rank - 1; i >= 0; --i) {
        m_offset[i] =
            (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i];
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    // Check if offset+tiledim in bounds - if not, replace tile dims with the
    // partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims, m_offset);

    Tile_Inner_Loop_Type<RP::rank, (RP::inner_direction == Iterate::Left), index_type,
                   Tag>::apply(m_func_innermost_loop, full_tile, m_offset, m_rp.m_tile,
                               m_tiledims);
  }

  index_type num_outer_iters() const { return m_num_outer_iters; }   

  template <typename... Args>
  std::enable_if_t<(sizeof...(Args) == RP::rank && std::is_void_v<Tag>), void>
  apply(Args&&... args) const {
    m_func(args...);
  }

  template <typename... Args>
  std::enable_if_t<(sizeof...(Args) == RP::rank && !std::is_void_v<Tag>), void>
  apply(Args&&... args) const {
    m_func(m_tag, args...);
  }

  RP const m_rp;
  Functor const m_func;
  index_type m_num_outer_iters = 1;
  std::conditional_t<std::is_void_v<Tag>, int, Tag> m_tag{};
};

// ------------------------------------------------------------------ //

#undef KOKKOS_IMPL_APPLY
#undef KOKKOS_IMPL_LOOP_R_1
#undef KOKKOS_IMPL_LOOP_R_2
#undef KOKKOS_IMPL_LOOP_R_3
#undef KOKKOS_IMPL_LOOP_R_4
#undef KOKKOS_IMPL_LOOP_R_5
#undef KOKKOS_IMPL_LOOP_R_6
#undef KOKKOS_IMPL_LOOP_R_7
#undef KOKKOS_IMPL_LOOP_R_8
#undef KOKKOS_IMPL_LOOP_L_1
#undef KOKKOS_IMPL_LOOP_L_2
#undef KOKKOS_IMPL_LOOP_L_3
#undef KOKKOS_IMPL_LOOP_L_4
#undef KOKKOS_IMPL_LOOP_L_5
#undef KOKKOS_IMPL_LOOP_L_6
#undef KOKKOS_IMPL_LOOP_L_7
#undef KOKKOS_IMPL_LOOP_L_8
#undef KOKKOS_IMPL_LOOP_LAYOUT_1
#undef KOKKOS_IMPL_LOOP_LAYOUT_2
#undef KOKKOS_IMPL_LOOP_LAYOUT_3
#undef KOKKOS_IMPL_LOOP_LAYOUT_4
#undef KOKKOS_IMPL_LOOP_LAYOUT_5
#undef KOKKOS_IMPL_LOOP_LAYOUT_6
#undef KOKKOS_IMPL_LOOP_LAYOUT_7
#undef KOKKOS_IMPL_LOOP_LAYOUT_8
#undef KOKKOS_IMPL_TILE_LOOP_1
#undef KOKKOS_IMPL_TILE_LOOP_2
#undef KOKKOS_IMPL_TILE_LOOP_3
#undef KOKKOS_IMPL_TILE_LOOP_4
#undef KOKKOS_IMPL_TILE_LOOP_5
#undef KOKKOS_IMPL_TILE_LOOP_6
#undef KOKKOS_IMPL_TILE_LOOP_7
#undef KOKKOS_IMPL_TILE_LOOP_8
#undef KOKKOS_IMPL_TAGGED_APPLY
#undef KOKKOS_IMPL_TAGGED_LOOP_R_1
#undef KOKKOS_IMPL_TAGGED_LOOP_R_2
#undef KOKKOS_IMPL_TAGGED_LOOP_R_3
#undef KOKKOS_IMPL_TAGGED_LOOP_R_4
#undef KOKKOS_IMPL_TAGGED_LOOP_R_5
#undef KOKKOS_IMPL_TAGGED_LOOP_R_6
#undef KOKKOS_IMPL_TAGGED_LOOP_R_7
#undef KOKKOS_IMPL_TAGGED_LOOP_R_8
#undef KOKKOS_IMPL_TAGGED_LOOP_L_1
#undef KOKKOS_IMPL_TAGGED_LOOP_L_2
#undef KOKKOS_IMPL_TAGGED_LOOP_L_3
#undef KOKKOS_IMPL_TAGGED_LOOP_L_4
#undef KOKKOS_IMPL_TAGGED_LOOP_L_5
#undef KOKKOS_IMPL_TAGGED_LOOP_L_6
#undef KOKKOS_IMPL_TAGGED_LOOP_L_7
#undef KOKKOS_IMPL_TAGGED_LOOP_L_8
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_1
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_2
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_3
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_4
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_5
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_6
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_7
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_8
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_1
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_2
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_3
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_4
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_5
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_6
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_7
#undef KOKKOS_IMPL_TAGGED_TILE_LOOP_8

}  // namespace Impl
}  // namespace Kokkos

#endif
