// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP
#define KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP

#include <type_traits>

#include <Kokkos_Layout.hpp>
#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP) && !defined(__CUDA_ARCH__)
#define KOKKOS_MDRANGE_IVDEP_INNERMOST_LOOP
#endif

#ifdef KOKKOS_MDRANGE_IVDEP_INNERMOST_LOOP
#if defined(__clang__)
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP \
  _Pragma("clang loop vectorize(assume_safety)")
#else
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP _Pragma("ivdep")
#endif
#else
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP
#endif

// FIXME: _Pragma("GCC ivdep") is preferable for GCC.
// However, for GCC < 11.5, spurious "warning: ignoring loop annotation"
// appears. Refer: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114691

namespace Kokkos {
namespace Impl {

// MDRangePolicy iteration via a nested loop without tiles

// Primary template for iterating via a nested loop without tiles
template <typename RP, typename Functor, typename Tag = void,
          typename ValueType = void, typename Enable = void>
struct HostIterateNestLoopWoTile;

// Currently, specialization for ParallelFor only
template <typename RP, typename Functor, typename Tag, typename ValueType>
struct HostIterateNestLoopWoTile<RP, Functor, Tag, ValueType,
                                 std::enable_if_t<std::is_void_v<ValueType>>> {
  using index_type = RP::index_type;

  inline HostIterateNestLoopWoTile(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  template <typename... Idxs>
  void func_innermost_loop(Idxs&&... idxs) const {
    if constexpr (RP::outer_direction == Iterate::Left) {
      KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP
      for (index_type i = m_rp.m_lower[0]; i < m_rp.m_upper[0]; ++i) {
        if constexpr (std::is_void_v<Tag>) {
          m_func(i, (Idxs&&)idxs...);
        } else {
          m_func(Tag{}, i, (Idxs&&)idxs...);
        }
      }
    } else {
      KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP
      for (index_type i = m_rp.m_lower[RP::rank - 1];
           i < m_rp.m_upper[RP::rank - 1]; ++i) {
        if constexpr (std::is_void_v<Tag>) {
          m_func((Idxs&&)idxs..., i);
        } else {
          m_func(Tag{}, (Idxs&&)idxs..., i);
        }
      }
    }
  }

  // ----------------------------------------------------------------------- //
  // \brief Nested loops with recursive template instantiation
  //
  // Nested for loop order depends on the iteration order:
  //  Iterate::Left:
  //    Outermost loop corresponds to right-most index.
  //  Iterate::Right:
  //    Outermost loop corresponds to left-most index.
  // The fastest changing index is always in innermost loop.
  //
  // Indices accumulated in parameter pack Idxs...
  //
  // Functor call order depends on the iteration order:
  //  Iterate::Left:
  //    functor(i_0, i_1, i_2, ..., i_{R-1})
  //  Iterate::Right:
  //    functor(i_{R-1}, ..., i_2, i_1, i_0)
  //
  //  \tparam IterLevel iteration level of the nested loops
  //  \tpraram Idxs... index pack
  template <unsigned IterLevel, typename... Idxs>
  inline void iterate(std::integral_constant<unsigned, IterLevel>,
                      Idxs... idxs) const {
    const index_type start = (RP::inner_direction == Iterate::Left)
                                 ? m_rp.m_lower[RP::rank - 1 - IterLevel]
                                 : m_rp.m_lower[IterLevel];
    const index_type end   = (RP::inner_direction == Iterate::Left)
                                 ? m_rp.m_upper[RP::rank - 1 - IterLevel]
                                 : m_rp.m_upper[IterLevel];

    for (index_type idx = start; idx < end; ++idx) {
      if constexpr (RP::inner_direction == Iterate::Left) {
        iterate(std::integral_constant<unsigned, IterLevel + 1>(), idx,
                idxs...);
      } else {
        iterate(std::integral_constant<unsigned, IterLevel + 1>(), idxs...,
                idx);
      }
    }
  }

  template <typename... Idxs>
  inline void iterate(std::integral_constant<unsigned, RP::rank - 1>,
                      Idxs... idxs) const {
    func_innermost_loop(idxs...);
  }

  inline void execute() const {
    iterate(std::integral_constant<unsigned, 0u>());
  }

  const RP m_rp;
  const Functor& m_func;
};

}  // namespace Impl
}  // namespace Kokkos

#endif
