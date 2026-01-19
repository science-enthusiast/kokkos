// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STD_ALGORITHMS_INCLUSIVE_SCAN_BY_KEY_HPP
#define KOKKOS_STD_ALGORITHMS_INCLUSIVE_SCAN_BY_KEY_HPP

#include <concepts>
#include <iostream>

#include "impl/Kokkos_InclusiveScanByKey.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

// Our APIs match those in the Thrust library by Nvidia and AMD.
// Particularly, we define a segment as a set of contiguous positions for
// which all the keys are the same.
//

//
// overload set accepting execution space
//

//
// overload set 1 default binary predicate for keys and default associative
// operator
//

template <typename ExecutionSpace, typename InputIterator1,
          typename InputIterator2, typename OutputIterator,
          std::enable_if_t<Impl::are_iterators_v<InputIterator1, InputIterator2,
                                                 OutputIterator>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
OutputIterator inclusive_scan_by_key(const ExecutionSpace& ex,
                                     InputIterator1 first_key,
                                     InputIterator1 last_key,
                                     InputIterator2 first_from,
                                     OutputIterator first_dest) {
  return Impl::in_scan_by_key_default_op_exespace_impl(
      "Kokkos::inclusive_scan_by_key_default_functors_iterator_api", ex,
      first_key, last_key, first_from, first_dest);
}

template <typename ExecutionSpace, typename InputIterator1,
          typename InputIterator2, typename OutputIterator,
          std::enable_if_t<Impl::are_iterators_v<InputIterator1, InputIterator2,
                                                 OutputIterator>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
OutputIterator inclusive_scan_by_key(const std::string& label,
                                     const ExecutionSpace& ex,
                                     InputIterator1 first_key,
                                     InputIterator1 last_key,
                                     InputIterator2 first_from,
                                     OutputIterator first_dest) {
  return Impl::in_scan_by_key_default_op_exespace_impl(
      label, ex, first_key, last_key, first_from, first_dest);
}

template <typename ExecutionSpace, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3,
          std::enable_if_t<std::equality_comparable<DataType1>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
auto inclusive_scan_by_key(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_default_op_exespace_impl(
      "Kokkos::inclusive_scan_by_key_default_functors_view_api", ex,
      KE::cbegin(view_key), KE::cend(view_key), KE::cbegin(view_from),
      KE::begin(view_dest));
}

template <typename ExecutionSpace, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3,
          std::enable_if_t<std::equality_comparable<DataType1>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
auto inclusive_scan_by_key(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_default_op_exespace_impl(
      label, ex, KE::cbegin(view_key), KE::cend(view_key),
      KE::cbegin(view_from), KE::begin(view_dest));
}

//
// overload set 2 custom binary predicate for keys
//

template <typename ExecutionSpace, typename InputIterator1,
          typename InputIterator2, typename OutputIterator, typename BinaryPred,
          std::enable_if_t<Impl::are_iterators_v<InputIterator1, InputIterator2,
                                                 OutputIterator>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
OutputIterator inclusive_scan_by_key(const ExecutionSpace& ex,
                                     InputIterator1 first_key,
                                     InputIterator1 last_key,
                                     InputIterator2 first_from,
                                     OutputIterator first_dest,
                                     BinaryPred binary_pred) {
  return Impl::in_scan_by_key_custom_binary_pred_exespace_impl(
      "Kokkos::inclusive_scan_custom_pred_iterator_api", ex, first_key,
      last_key, first_from, first_dest, binary_pred);
}

template <typename ExecutionSpace, typename InputIterator1,
          typename InputIterator2, typename OutputIterator, typename BinaryPred,
          std::enable_if_t<Impl::are_iterators_v<InputIterator1, InputIterator2,
                                                 OutputIterator>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
OutputIterator inclusive_scan_by_key(const std::string& label,
                                     const ExecutionSpace& ex,
                                     InputIterator1 first_key,
                                     InputIterator1 last_key,
                                     InputIterator2 first_from,
                                     OutputIterator first_dest,
                                     BinaryPred binary_pred) {
  return Impl::in_scan_by_key_custom_binary_pred_exespace_impl(
      label, ex, first_key, last_key, first_from, first_dest, binary_pred);
}

template <typename ExecutionSpace, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3, typename BinaryPred,
          std::enable_if_t<std::equality_comparable<DataType1>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
auto inclusive_scan_by_key(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest,
    BinaryPred binary_pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_custom_binary_pred_exespace_impl(
      "Kokkos::inclusive_scan_custom_pred_view_api", ex, KE::cbegin(view_key),
      KE::cend(view_key), KE::cbegin(view_from), KE::begin(view_dest),
      binary_pred);
}

template <typename ExecutionSpace, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3, typename BinaryPred,
          std::enable_if_t<std::equality_comparable<DataType1>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
auto inclusive_scan_by_key(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest,
    BinaryPred binary_pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_custom_binary_pred_exespace_impl(
      label, ex, KE::cbegin(view_key), KE::cend(view_key),
      KE::cbegin(view_from), KE::begin(view_dest), binary_pred);
}

//
// overload set 3 custom binary predicate for keys and associative operator
//

template <typename ExecutionSpace, typename InputIterator1,
          typename InputIterator2, typename OutputIterator, typename BinaryPred,
          typename AssociativeOp,
          std::enable_if_t<Impl::are_iterators_v<InputIterator1, InputIterator2,
                                                 OutputIterator>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
OutputIterator inclusive_scan_by_key(
    const ExecutionSpace& ex, InputIterator1 first_key, InputIterator1 last_key,
    InputIterator2 first_from, OutputIterator first_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  return Impl::in_scan_by_key_custom_pred_op_exespace_impl(
      "Kokkos::inclusive_scan_custom_pred_op_iterator_api", ex, first_key,
      last_key, first_from, first_dest, binary_pred, binary_op);
}

template <typename ExecutionSpace, typename InputIterator1,
          typename InputIterator2, typename OutputIterator, typename BinaryPred,
          typename AssociativeOp,
          std::enable_if_t<Impl::are_iterators_v<InputIterator1, InputIterator2,
                                                 OutputIterator>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
OutputIterator inclusive_scan_by_key(
    const std::string& label, const ExecutionSpace& ex,
    InputIterator1 first_key, InputIterator1 last_key,
    InputIterator2 first_from, OutputIterator first_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  return Impl::in_scan_by_key_custom_pred_op_exespace_impl(
      label, ex, first_key, last_key, first_from, first_dest, binary_pred,
      binary_op);
}

template <typename ExecutionSpace, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3, typename BinaryPred, typename AssociativeOp,
          std::enable_if_t<std::equality_comparable<DataType1>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
auto inclusive_scan_by_key(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_custom_pred_op_exespace_impl(
      "Kokkos::inclusive_scan_custom_pred_op_view_api", ex,
      KE::cbegin(view_key), KE::cend(view_key), KE::cbegin(view_from),
      KE::begin(view_dest), binary_pred, binary_op);
}

template <typename ExecutionSpace, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3, typename BinaryPred, typename AssociativeOp,
          std::enable_if_t<std::equality_comparable<DataType1>&& ::Kokkos::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
auto inclusive_scan_by_key(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_custom_pred_op_exespace_impl(
      label, ex, KE::cbegin(view_key), KE::cend(view_key),
      KE::cbegin(view_from), KE::begin(view_dest), binary_pred, binary_op);
}

//
// overload set accepting team handle
//
// Note: Due to the same reasoning as other std_algorithms functions
// the overloads accepting a label have been omitted since there is
// the issue with GPUs and std::strings.
//
// Note: The call to inclusive_scan_by_key happens based on a TeamThreadRange
// (a nested policy). At the same time, the keys have to be transformed into
// head flag based representation of the segments, internally. Also, as an
// internal detail, these head flags are expected to be of boolean type. Thus, a
// View of (bool, value) pairs needs to be created and passed before the Team
// Policy based parallel_for kernel is launched. This View should have the same
// dimensions as the other Views (keys, input values, output values) and it will
// be used internally.
// Refer to algorithms/unit_tests/TestStdAlgorithmsTeamInclusiveScanByKey.cpp
// for example usage.

//
// overload set 1 default associative operator
//

template <typename TeamHandle, typename InputIterator1, typename InputIterator2,
          typename BoolKeyValueIterator, typename OutputIterator,
          std::enable_if_t<
              Impl::are_iterators_v<
                  InputIterator1, InputIterator2, BoolKeyValueIterator,
                  OutputIterator>&& ::Kokkos::is_team_handle_v<TeamHandle>,
              int> = 0>
KOKKOS_FUNCTION OutputIterator inclusive_scan_by_key(
    const TeamHandle& team_handle, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    BoolKeyValueIterator first_bool_key_value, OutputIterator first_dest) {
  return Impl::in_scan_by_key_default_op_team_impl(
      team_handle, first_key, last_key, first_from, first_bool_key_value,
      first_dest);
}

template <typename TeamHandle, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3, typename DataType4, typename... Properties4,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandle>, int> = 0>
KOKKOS_FUNCTION auto inclusive_scan_by_key(
    const TeamHandle& team_handle,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_bool_key_value,
    const ::Kokkos::View<DataType4, Properties4...>& view_dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(
      view_bool_key_value);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_default_op_team_impl(
      team_handle, KE::cbegin(view_key), KE::cend(view_key),
      KE::cbegin(view_from), KE::begin(view_bool_key_value),
      KE::begin(view_dest));
}

//
// overload set 2 custom binary predicate for keys
//

template <typename TeamHandle, typename InputIterator1, typename InputIterator2,
          typename BoolKeyValueIterator, typename OutputIterator,
          typename BinaryPred,
          std::enable_if_t<
              Impl::are_iterators_v<
                  InputIterator1, InputIterator2, BoolKeyValueIterator,
                  OutputIterator>&& ::Kokkos::is_team_handle_v<TeamHandle>,
              int> = 0>
KOKKOS_FUNCTION OutputIterator
inclusive_scan_by_key(const TeamHandle& team_handle, InputIterator1 first_key,
                      InputIterator1 last_key, InputIterator2 first_from,
                      BoolKeyValueIterator first_bool_key_value,
                      OutputIterator first_dest, BinaryPred binary_pred) {
  return Impl::in_scan_by_key_custom_binary_pred_team_impl(
      team_handle, first_key, last_key, first_from, first_bool_key_value,
      first_dest, binary_pred);
}

template <typename TeamHandle, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3, typename DataType4, typename... Properties4,
          typename BinaryPred,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandle>, int> = 0>
KOKKOS_FUNCTION auto inclusive_scan_by_key(
    const TeamHandle& team_handle,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_bool_key_value,
    const ::Kokkos::View<DataType4, Properties4...>& view_dest,
    BinaryPred binary_pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(
      view_bool_key_value);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_custom_binary_pred_team_impl(
      team_handle, KE::cbegin(view_key), KE::cend(view_key),
      KE::cbegin(view_from), KE::begin(view_bool_key_value),
      KE::begin(view_dest), binary_pred);
}

//
// overload set 3 custom binary predicate for keys and custom associative
// operator
//

template <typename TeamHandle, typename InputIterator1, typename InputIterator2,
          typename BoolKeyValueIterator, typename OutputIterator,
          typename BinaryPred, typename AssociativeOp,
          std::enable_if_t<
              Impl::are_iterators_v<
                  InputIterator1, InputIterator2, BoolKeyValueIterator,
                  OutputIterator>&& ::Kokkos::is_team_handle_v<TeamHandle>,
              int> = 0>
KOKKOS_FUNCTION OutputIterator inclusive_scan_by_key(
    const TeamHandle& team_handle, InputIterator1 first_key,
    InputIterator1 last_key, InputIterator2 first_from,
    BoolKeyValueIterator first_bool_key_value, OutputIterator first_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  return Impl::in_scan_by_key_custom_pred_op_team_impl(
      team_handle, first_key, last_key, first_from, first_bool_key_value,
      first_dest, binary_pred, binary_op);
}

template <typename TeamHandle, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename DataType3,
          typename... Properties3, typename DataType4, typename... Properties4,
          typename BinaryPred, typename AssociativeOp,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandle>, int> = 0>
KOKKOS_FUNCTION auto inclusive_scan_by_key(
    const TeamHandle& team_handle,
    const ::Kokkos::View<DataType1, Properties1...>& view_key,
    const ::Kokkos::View<DataType2, Properties2...>& view_from,
    const ::Kokkos::View<DataType3, Properties3...>& view_bool_key_value,
    const ::Kokkos::View<DataType4, Properties4...>& view_dest,
    BinaryPred binary_pred, AssociativeOp binary_op) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_key);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(
      view_bool_key_value);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::in_scan_by_key_custom_pred_op_team_impl(
      team_handle, KE::cbegin(view_key), KE::cend(view_key),
      KE::cbegin(view_from), KE::begin(view_bool_key_value),
      KE::begin(view_dest), binary_pred, binary_op);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
