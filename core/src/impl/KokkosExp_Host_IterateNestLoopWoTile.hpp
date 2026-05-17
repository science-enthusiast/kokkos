// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP
#define KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP

#include <type_traits>

#include <Kokkos_Layout.hpp>
#include <Kokkos_Macros.hpp>

namespace Kokkos {
namespace Impl {

template <class Tag, class Functor, class... Args>
KOKKOS_IMPL_FORCEINLINE_FUNCTION std::enable_if_t<std::is_void_v<Tag>>
_host_tag_invoke(Functor const& f, Args&&... args) {
  f((Args&&)args...);
}

template <class Tag, class Functor, class... Args>
KOKKOS_IMPL_FORCEINLINE_FUNCTION std::enable_if_t<!std::is_void_v<Tag>>
_host_tag_invoke(Functor const& f, Args&&... args) {
  f(Tag{}, (Args&&)args...);
}

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

  // ----------------------------------------------------------------------- //
  // Nested loops with recursive template instantiation
  //
  // Accumulates indices in parameter pack Idxs...
  // The fastest changing index is always i0 (innermost loop).
  //
  // Functor call order depends on the iteration order:
  //  Iterate::Left:
  //    functor(i0, i1, i2, ..., iR)
  //  Iterate::Right:
  //    functor(iR, ..., i2, i1, i0)
  //
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
  inline void iterate(std::integral_constant<unsigned, RP::rank>,
                      Idxs... idxs) const {
    Impl::_host_tag_invoke<Tag>(m_func, idxs...);
  }

  inline void execute() const {
    if (m_rp.m_num_tiles > 0) {
      iterate(std::integral_constant<unsigned, 0u>());
    }
  }

  const RP m_rp;
  const Functor m_func;
};

}  // namespace Impl
}  // namespace Kokkos

#endif
