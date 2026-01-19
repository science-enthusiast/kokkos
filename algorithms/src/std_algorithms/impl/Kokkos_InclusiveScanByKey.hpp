// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STD_ALGORITHMS_INCLUSIVE_SCAN_BY_KEY_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_INCLUSIVE_SCAN_BY_KEY_IMPL_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_TransformInclusiveScan.hpp>
#include <std_algorithms/Kokkos_Distance.hpp>

#include <iostream>
#include <string>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)

// Workaround for `Instruction 'shfl' without '.sync' is not supported on
// .target sm_70 and higher from PTX ISA version 6.4`.
// Also see https://github.com/NVIDIA/cub/pull/170.
#if !defined(CUB_USE_COOPERATIVE_GROUPS)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsuggest-override"

#if CUDA_VERSION >= 13010
#include <cuda/std/iterator>
#else
#include <thrust/distance.h>
#endif
#include <thrust/scan.h>

#pragma GCC diagnostic pop

#elif defined(KOKKOS_ENABLE_ROCTHRUST)

#include <thrust/distance.h>
#include <thrust/scan.h>

#endif

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ValueType>
struct EqualityFunctor {
  using value_type = ValueType;

  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType& a, const ValueType& b) const {
    return a == b;
  }
};

template <class ValueType>
struct SumFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a + b);
  }
};

namespace KE = Kokkos::Experimental;

//
// An operator transformation based approach for segmented scan.
// Refer to Section 1.5 of Prefix Sums and Their Applications
// by Guy Blelloch, 1990. For simpler notation, refer to
// section 2.1 of Efficient Parallel Scan Algorithms for GPUs,
// S Sengupta et al, 2010.
//

// In this approach, given a key View and a value View,
// a View is prepared, which will contain elements that are each a pair of a
// boolean key and the corresponding input value.
// TransformInclusiveScan is performed on this View of pairs by using a
// special binary predicate (see struct InScanByKeyBinaryOp below).
// The operator transformation approach expects a true head flag at the start
// of a segment and false in the rest of the positions.
// However, in our user facing API, we define a segment as a set of contiguous
// positions that have the same keys. Internally we transform that notation to
// head flag based representation. Refer the functor FillBoolKeyValueView.

template <class KeyType, class ValueType>
struct KeyValuePair {
  using key_type   = KeyType;
  using value_type = ValueType;

  KOKKOS_INLINE_FUNCTION
  KeyValuePair& operator+=(const KeyValuePair& kv_pair) {
    key |= kv_pair.key;      // assume |= is defined for key_type
    if (kv_pair.key)
      value = kv_pair.value;
    else
      value += kv_pair.value;  // assume += is defined for value_type
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  KeyValuePair() = default;

  KOKKOS_INLINE_FUNCTION
  KeyValuePair(KeyType k, ValueType v) : key(k), value(v) {}

  // A copy constructor which takes a const volatile qualified
  // argument and a copy assignment operator that is volatile
  // qualified are needed because of how parallel_scan is
  // implemented in core/src/Threads/Kokkos_Threads_Team.hpp.
  // The variable work_value is a pointer to volatile qualified value type.
  // The rest of the special member functions have been explicitly
  // defined due to Rule of Five and to enable move semantics.

  KOKKOS_INLINE_FUNCTION
  KeyValuePair(const volatile KeyValuePair& other)
      : key(other.key), value(other.value) {}

  KOKKOS_INLINE_FUNCTION
  KeyValuePair(KeyValuePair&& other) = default;

  KOKKOS_INLINE_FUNCTION
  void operator=(const volatile KeyValuePair& other) volatile {
    this->key   = other.key;
    this->value = other.value;
  }

  KOKKOS_INLINE_FUNCTION
  KeyValuePair& operator=(KeyValuePair&& other) = default;

  KOKKOS_INLINE_FUNCTION
  ~KeyValuePair() = default;

  KeyType key;
  ValueType value;
};

template <class KeyValuePair>
KOKKOS_INLINE_FUNCTION KeyValuePair operator+(KeyValuePair kv_pair_1,
                                              const KeyValuePair& kv_pair_2) {
  return kv_pair_1 += kv_pair_2;
}

template <class KeyIterator, class ValueIterator, class BoolKeyValueIterator,
          class BinaryPred>
struct FillBoolKeyValueView {
  KeyIterator m_first_key;
  ValueIterator m_first_from;
  BoolKeyValueIterator m_first_bool_key_value;
  BinaryPred m_binary_pred;

  KOKKOS_FUNCTION
  FillBoolKeyValueView(KeyIterator first_key, ValueIterator first_from,
                       BoolKeyValueIterator first_bool_key_value,
                       BinaryPred binary_pred)
      : m_first_key(std::move(first_key)),
        m_first_from(std::move(first_from)),
        m_first_bool_key_value(std::move(first_bool_key_value)),
        m_binary_pred(std::move(binary_pred)) {}

  KOKKOS_FUNCTION
  void operator()(const int i) const {
    if (m_binary_pred(m_first_key[i], m_first_key[i - 1])) {
      m_first_bool_key_value[i].key = false;
    } else {
      m_first_bool_key_value[i].key = true;
    }
    m_first_bool_key_value[i].value = m_first_from[i];
  }
};

// The following binary operator implements how key-value pairs are combined
// in segmented scan. This binary operator is passed to the functor that
// is already implemented in Kokko_TransformInclusiveScan.hpp

template <class KeyValueType, class AssociativeOp>
struct InScanByKeyBinaryOp {
  AssociativeOp m_binary_op;

  KOKKOS_INLINE_FUNCTION
  InScanByKeyBinaryOp(AssociativeOp binary_op)
      : m_binary_op(std::move(binary_op)) {}

  KOKKOS_INLINE_FUNCTION
  KeyValueType operator()(const KeyValueType& x, const KeyValueType& y) const {
    return {x.key || y.key, (y.key) ? y.value : m_binary_op(x.value, y.value)};
  }
};

// -----------------------------------------------------------------------------
// inclusive scan by key: custom predicate for keys and associative operator
// -----------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class InputIterator1, class InputIterator2, class OutputIterator,
          class BinaryPred, class AssociativeOp>
OutputIterator in_scan_by_key_custom_pred_op_exespace_impl(
    const std::string& label, const Cuda& ex, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    OutputIterator first_dest, BinaryPred binary_pred,
    AssociativeOp binary_op) {
  const auto thrust_ex = thrust::cuda::par.on(ex.cuda_stream());

  Kokkos::Profiling::pushRegion(label + " via thrust::inclusive_scan_by_key");

  thrust::inclusive_scan_by_key(thrust_ex, first_key, last_key, first_from,
                                first_dest, binary_pred, binary_op);

  Kokkos::Profiling::popRegion();

#if CUDA_VERSION >= 13010
  const auto num_elements = cuda::std::distance(first_key, last_key);
#else
  const auto num_elements = thrust::distance(first_key, last_key);
#endif

  return first_dest + num_elements;
}

#elif defined(KOKKOS_ENABLE_ROCTHRUST)
template <class InputIterator1, class InputIterator2, class OutputIterator,
          class BinaryPred, class AssociativeOp>
OutputIterator in_scan_by_key_custom_pred_op_exespace_impl(
    const std::string& label, const HIP& ex, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    OutputIterator first_dest, BinaryPred binary_pred,
    AssociativeOp binary_op) {
  const auto thrust_ex = thrust::hip::par.on(ex.hip_stream());

  Kokkos::Profiling::pushRegion(label + " via thrust::inclusive_scan_by_key");

  thrust::inclusive_scan_by_key(thrust_ex, first_key, last_key, first_from,
                                first_dest, binary_pred, binary_op);

  Kokkos::Profiling::popRegion();

  const auto num_elements = thrust::distance(first_from, last_from);

  return first_dest + num_elements;
}
#endif

template <class ExecutionSpace, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryPred, class AssociativeOp>
OutputIterator in_scan_by_key_custom_pred_op_exespace_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIterator1 first_key, InputIterator1 last_key,
    InputIterator2 first_from, OutputIterator first_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first_key, first_from,
                                                   first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_key, last_key);

  using index_type    = typename InputIterator2::difference_type;
  using unary_op_type = StdNumericScanIdentityReferenceUnaryFunctor;
  using key_type   = std::remove_const_t<typename InputIterator1::value_type>;
  using value_type = std::remove_const_t<typename InputIterator2::value_type>;
  using bool_key_value_type = KeyValuePair<bool, value_type>;

  Kokkos::Profiling::pushRegion(label + " via Kokkos::parallel_scan");

  const auto num_elements = KE::distance(first_key, last_key);

  Kokkos::View<bool_key_value_type*, ExecutionSpace> bool_key_value_view(
      "bool_key_value_view", num_elements);

  auto first_bool_key_value = KE::begin(bool_key_value_view);

  using fill_bool_key_value_func_type =
      FillBoolKeyValueView<InputIterator1, InputIterator2,
                           decltype(first_bool_key_value), BinaryPred>;
  fill_bool_key_value_func_type fill_bool_key_value_func(
      first_key, first_from, first_bool_key_value, binary_pred);

  if (num_elements > 0) {
    first_bool_key_value[0].key   = true;
    first_bool_key_value[0].value = first_from[0];

    Kokkos::parallel_for("fill_bool_key_value_view",
                         Kokkos::RangePolicy(ex, 1, num_elements),
                         fill_bool_key_value_func);
    Kokkos::fence();
  }

  using kv_iterator_type = decltype(KE::begin(bool_key_value_view));
  using transform_binary_op_type =
      InScanByKeyBinaryOp<bool_key_value_type, AssociativeOp>;

  auto transform_binary_op = transform_binary_op_type(binary_op);

  using func_type = ExeSpaceTransformInclusiveScanNoInitValueFunctor<
      ExecutionSpace, index_type, bool_key_value_type, kv_iterator_type,
      kv_iterator_type, transform_binary_op_type, unary_op_type>;

  Kokkos::parallel_scan(
      label, RangePolicy(ex, 0, num_elements),
      func_type(KE::begin(bool_key_value_view), KE::begin(bool_key_value_view),
                transform_binary_op, unary_op_type()));

  Kokkos::fence();

  Kokkos::parallel_for(
      "extract_scan_values", RangePolicy(ex, 0, num_elements),
      KOKKOS_LAMBDA(const int i) {
        first_dest[i] = bool_key_value_view(i).value;
      });

  Kokkos::fence();

  Kokkos::Profiling::popRegion();

  return first_dest + num_elements;
}

// -------------------------------------------------------------
// inclusive scan by key: default functors
// -------------------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class InputIterator1, class InputIterator2, class OutputIterator>
OutputIterator in_scan_by_key_default_op_exespace_impl(
    const std::string& label, const Cuda& ex, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    OutputIterator first_dest) {
  const auto thrust_ex = thrust::cuda::par.on(ex.cuda_stream());

  Kokkos::Profiling::pushRegion(label + " via thrust::inclusive_scan_by_key");

  thrust::inclusive_scan_by_key(thrust_ex, first_key, last_key, first_from,
                                first_dest);

  Kokkos::Profiling::popRegion();

#if CUDA_VERSION >= 13010
  const auto num_elements = cuda::std::distance(first_key, last_key);
#else
  const auto num_elements = thrust::distance(first_key, last_key);
#endif

  return first_dest + num_elements;
}

#elif defined(KOKKOS_ENABLE_ROCTHRUST)
template <class InputIterator1, class InputIterator2, class OutputIterator>
OutputIterator in_scan_by_key_default_op_exespace_impl(
    const std::string& label, const HIP& ex, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    OutputIterator first_dest) {
  const auto thrust_ex = thrust::hip::par.on(ex.hip_stream());

  Kokkos::Profiling::pushRegion(label + " via thrust::inclusive_scan_by_key");

  thrust::inclusive_scan_by_key(thrust_ex, first_key, last_key, first_from,
                                first_dest);

  Kokkos::Profiling::popRegion();

  const auto num_elements = thrust::distance(first_key, last_key);

  return first_dest + num_elements;
}

#endif

template <class ExecutionSpace, class InputIterator1, class InputIterator2,
          class OutputIterator>
OutputIterator in_scan_by_key_default_op_exespace_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIterator1 first_key, InputIterator1 last_key,
    InputIterator2 first_from, OutputIterator first_dest) {
  using key_type   = std::remove_const_t<typename InputIterator1::value_type>;
  using value_type = std::remove_const_t<typename InputIterator2::value_type>;

  return in_scan_by_key_custom_pred_op_exespace_impl(
      label, ex, first_key, last_key, first_from, first_dest,
      EqualityFunctor<key_type>(), SumFunctor<value_type>());
}

// -------------------------------------------------------------
// inclusive scan by key: custom binary predicate for keys
// -------------------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class InputIterator1, class InputIterator2, class OutputIterator,
          class BinaryPred>
OutputIterator in_scan_by_key_custom_binary_pred_exespace_impl(
    const std::string& label, const Cuda& ex, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    OutputIterator first_dest, BinaryPred binary_pred) {
  const auto thrust_ex = thrust::cuda::par.on(ex.cuda_stream());

  Kokkos::Profiling::pushRegion(label + " via thrust::inclusive_scan_by_key");

  thrust::inclusive_scan_by_key(thrust_ex, first_key, last_key, first_from,
                                first_dest, binary_pred);

  Kokkos::Profiling::popRegion();

#if CUDA_VERSION >= 13010
  const auto num_elements = cuda::std::distance(first_key, last_key);
#else
  const auto num_elements = thrust::distance(first_key, last_key);
#endif

  return first_dest + num_elements;
}

#elif defined(KOKKOS_ENABLE_ROCTHRUST)
template <class InputIterator1, class InputIterator2, class OutputIterator,
          class BinaryPred>
OutputIterator in_scan_by_key_custom_binary_pred_exespace_impl(
    const std::string& label, const HIP& ex, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    OutputIterator first_dest, BinaryPred binary_pred) {
  const auto thrust_ex = thrust::hip::par.on(ex.hip_stream());

  Kokkos::Profiling::pushRegion(label + " via thrust::inclusive_scan_by_key");

  thrust::inclusive_scan_by_key(thrust_ex, first_key, last_key, first_from,
                                first_dest, binary_pred);

  Kokkos::Profiling::popRegion();

  const auto num_elements = thrust::distance(first_key, last_key);

  return first_dest + num_elements;
}
#endif

template <class ExecutionSpace, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryPred>
OutputIterator in_scan_by_key_custom_binary_pred_exespace_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIterator1 first_key, InputIterator1 last_key,
    InputIterator2 first_from, OutputIterator first_dest,
    BinaryPred binary_pred) {
  using value_type = std::remove_const_t<typename InputIterator2::value_type>;

  return in_scan_by_key_custom_pred_op_exespace_impl(
      label, ex, first_key, last_key, first_from, first_dest,
      EqualityFunctor<bool>(), SumFunctor<value_type>());
}

//
// Team overloads
//

// -----------------------------------------------------------------------------
// inclusive scan by key: Team execution policy,
//                        custom predicate for keys and associative operator
// -----------------------------------------------------------------------------

template <class TeamHandle, class InputIterator1, class InputIterator2,
          class BoolKeyValueIterator, class OutputIterator, class BinaryPred,
          class AssociativeOp>
KOKKOS_FUNCTION OutputIterator in_scan_by_key_custom_pred_op_team_impl(
    const TeamHandle& team_handle, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    BoolKeyValueIterator first_bool_key_value, OutputIterator first_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  // checks
  Impl::static_assert_random_access_and_accessible(
      team_handle, first_key, first_from, first_bool_key_value, first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_key, last_key);

  using exe_space     = typename TeamHandle::execution_space;
  using unary_op_type = StdNumericScanIdentityReferenceUnaryFunctor;
  using value_type = std::remove_const_t<typename InputIterator2::value_type>;
  using bool_key_value_type =
      BoolKeyValueIterator::value_type;  // Internally key type is always bool

  const auto num_elements = KE::distance(first_key, last_key);

  using fill_bool_key_value_func_type =
      FillBoolKeyValueView<InputIterator1, InputIterator2,
                           decltype(first_bool_key_value), BinaryPred>;
  fill_bool_key_value_func_type fill_bool_key_value_func(
      first_key, first_from, first_bool_key_value, binary_pred);

  if (num_elements > 0) {
    first_bool_key_value[0].key   = true;
    first_bool_key_value[0].value = first_from[0];

    Kokkos::parallel_for(TeamThreadRange(team_handle, 1, num_elements),
                         fill_bool_key_value_func);
    team_handle.team_barrier();
  }

  using transform_binary_op_type =
      InScanByKeyBinaryOp<bool_key_value_type, AssociativeOp>;

  auto transform_binary_op = transform_binary_op_type(binary_op);

  using func_type = TeamTransformInclusiveScanNoInitValueFunctor<
      exe_space, bool_key_value_type, BoolKeyValueIterator,
      BoolKeyValueIterator, transform_binary_op_type, unary_op_type>;

  Kokkos::parallel_scan(TeamThreadRange(team_handle, 0, num_elements),
                        func_type(first_bool_key_value, first_bool_key_value,
                                  binary_op, unary_op_type()));

  team_handle.team_barrier();

  Kokkos::parallel_for(
      TeamThreadRange(team_handle, 0, num_elements),
      KOKKOS_LAMBDA(const int i) {
        first_dest[i] = first_bool_key_value[i].value;
      });

  team_handle.team_barrier();

  return first_dest + num_elements;
}

// --------------------------------------------------------------
// inclusive scan by key: Team execution policy, default functors
// --------------------------------------------------------------
template <class TeamHandle, class InputIterator1, class InputIterator2,
          class BoolKeyValueIterator, class OutputIterator>
KOKKOS_FUNCTION OutputIterator in_scan_by_key_default_op_team_impl(
    const TeamHandle& team_handle, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    BoolKeyValueIterator first_bool_key_value, OutputIterator first_dest) {
  using key_type   = std::remove_const_t<typename InputIterator1::value_type>;
  using value_type = std::remove_const_t<typename InputIterator2::value_type>;

  return in_scan_by_key_custom_pred_op_team_impl(
      team_handle, first_key, last_key, first_from, first_bool_key_value,
      first_dest, EqualityFunctor<key_type>(), SumFunctor<value_type>());
}

// -------------------------------------------------------------
// inclusive scan by key: Team execution policy,
//                        custom binary predicate for keys
// -------------------------------------------------------------
template <class TeamHandle, class InputIterator1, class InputIterator2,
          class BoolKeyValueIterator, class OutputIterator, class BinaryPred>
KOKKOS_FUNCTION OutputIterator in_scan_by_key_custom_binary_pred_team_impl(
    const TeamHandle& team_handle, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    BoolKeyValueIterator first_bool_key_value, OutputIterator first_dest,
    BinaryPred binary_pred) {
  using value_type = std::remove_const_t<typename InputIterator2::value_type>;

  return in_scan_by_key_custom_pred_op_team_impl(
      team_handle, first_key, last_key, first_from, first_bool_key_value,
      first_dest, binary_pred, SumFunctor<value_type>());
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
