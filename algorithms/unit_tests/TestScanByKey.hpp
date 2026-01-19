// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TEST_SCAN_BY_KEY_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TEST_SCAN_BY_KEY_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestStdAlgorithmsCommon.hpp>

#include <iterator>

namespace Test {
namespace ScanByKey {

template <class ValueType>
struct EqualityFunctor {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType& a, const ValueType& b) const {
    return (a == b);
  }

  using value_type = ValueType;
};

template <class ValueType>
struct MultiplyFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a * b);
  }
};

template <class ValueType>
struct SumFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a + b);
  }
};

template <class InputIterator1, class InputIterator2, class OutputIterator,
          class BinaryPred, class AssociativeOp>
OutputIterator ref_scan_by_key(InputIterator1 key, InputIterator1 key_end,
                               InputIterator2 in, OutputIterator out,
                               BinaryPred binary_pred,
                               AssociativeOp binary_op) {
  std::size_t num_elems = std::distance(key, key_end);

  out[0] = in[0];

  for (std::size_t i = 1; i < num_elems; ++i) {
    if (binary_pred(key[i], key[i - 1])) {
      out[i] = binary_op(in[i], out[i - 1]);
    } else {
      out[i] = in[i];
    }
  }

  return out + num_elems;
}

template <class ViewType>
void fill_key_view(ViewType key_view) {
  using value_type = typename ViewType::value_type;
  using exe_space  = typename ViewType::execution_space;

  const std::size_t ext = key_view.extent(0);
  using aux_view_t      = Kokkos::View<value_type*, exe_space>;
  aux_view_t aux_view("aux_key_view", ext);
  auto v_h = create_mirror_view(Kokkos::HostSpace(), aux_view);

  if (ext < 4) {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(1);
    }
  } else {
    std::size_t one_fourth = ext / 4;

    for (std::size_t i = 0; i < one_fourth; ++i) {
      v_h(i)                  = static_cast<value_type>(1);
      v_h(one_fourth + i)     = static_cast<value_type>(0);
      v_h(2 * one_fourth + i) = static_cast<value_type>(1);
    }

    for (std::size_t i = 3 * one_fourth; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(0);
    }
  }

  Kokkos::deep_copy(aux_view, v_h);
  Test::stdalgos::CopyFunctor<aux_view_t, ViewType> F1(aux_view, key_view);
  Kokkos::parallel_for("copy", key_view.extent(0), F1);
}

template <class ValueType, class LayoutTagType>
auto create_2d_key_view_and_host_clone(LayoutTagType layout_tag,
                                       std::size_t num_rows,
                                       std::size_t num_cols,
                                       const std::string& label) {
  // construct in memory space associated with default exespace
  auto key_view = create_view<ValueType>(layout_tag, num_rows, num_cols, label);

  // key_view might not deep copyable (e.g. strided layout), we make a new
  // view that is for sure deep copyable, modify it on the host, deep copy
  // to device and then launch a kernel to copy to key_view
  auto key_view_dc =
      Test::stdalgos::create_deep_copyable_compatible_view_with_same_extent(
          key_view);

  auto key_view_dc_h =
      create_mirror_view(Kokkos::DefaultHostExecutionSpace(), key_view_dc);

  Kokkos::parallel_for(
      Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, num_rows),
      [=](int i) {
        if (num_cols < 4) {
          for (std::size_t j = 0; j < num_cols; ++j) {
            key_view_dc_h(i, j) = static_cast<ValueType>(1);
          }
        } else {
          std::size_t one_fourth = num_cols / 4;

          for (std::size_t j = 0; j < one_fourth; ++j) {
            key_view_dc_h(i, j)                  = static_cast<ValueType>(1);
            key_view_dc_h(i, one_fourth + j)     = static_cast<ValueType>(0);
            key_view_dc_h(i, 2 * one_fourth + j) = static_cast<ValueType>(1);
          }

          for (std::size_t j = 3 * one_fourth; j < num_cols; ++j) {
            key_view_dc_h(i, j) = static_cast<ValueType>(0);
          }
        }
      });

  Kokkos::deep_copy(key_view_dc, key_view_dc_h);

  Test::stdalgos::CopyFunctorRank2 F1(key_view_dc, key_view);
  Kokkos::parallel_for("copy_key_rows", key_view.extent(0) * key_view.extent(1),
                       F1);

  return std::make_pair(key_view, key_view_dc_h);
}

}  // namespace ScanByKey
}  // namespace Test

#endif
