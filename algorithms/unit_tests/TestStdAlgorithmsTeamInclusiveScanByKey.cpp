// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <iostream>
#include <vector>

#include <TestStdAlgorithmsCommon.hpp>
#include <TestScanByKey.hpp>
#include "std_algorithms/Kokkos_BeginEnd.hpp"

namespace Test {
namespace stdalgos {
namespace TeamInclusiveScanByKey {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct EqualityFunctor {
  KOKKOS_INLINE_FUNCTION
  constexpr bool operator()(const ValueType& a, const ValueType& b) const {
    return a == b;
  }
};

template <class ValueType>
struct PlusFunctor {
  KOKKOS_INLINE_FUNCTION
  constexpr ValueType operator()(const ValueType& a, const ValueType& b) const {
    return a + b;
  }
};

template <class KeyView, class SourceView, class BoolKeyValueView,
          class DestView, class DistancesView, class IntraTeamSentinelView>
struct TestFunctorA {
  KeyView m_key_view;
  SourceView m_src_view;
  BoolKeyValueView m_bool_key_value_view;
  DestView m_dst_view;
  DistancesView m_dists_view;
  IntraTeamSentinelView m_intra_team_sentinel_view;
  int m_api_pick;

  TestFunctorA(const KeyView key_view, const SourceView src_view,
               const BoolKeyValueView bool_key_value_view,
               const DestView dst_view, const DistancesView dists_view,
               const IntraTeamSentinelView intra_team_sentinel_view,
               int api_pick)
      : m_key_view(key_view),
        m_src_view(src_view),
        m_bool_key_value_view(bool_key_value_view),
        m_dst_view(dst_view),
        m_dists_view(dists_view),
        m_intra_team_sentinel_view(intra_team_sentinel_view),
        m_api_pick(api_pick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto row_index = member.league_rank();

    auto key_row = Kokkos::subview(m_key_view, row_index, Kokkos::ALL());
    const auto first_key = KE::cbegin(key_row);
    const auto last_key  = KE::cend(key_row);
    auto src_row = Kokkos::subview(m_src_view, row_index, Kokkos::ALL());
    const auto first_src = KE::begin(src_row);
    auto bool_key_value_row =
        Kokkos::subview(m_bool_key_value_view, row_index, Kokkos::ALL());
    const auto first_bool_key_value = KE::begin(bool_key_value_row);
    auto dst_row = Kokkos::subview(m_dst_view, row_index, Kokkos::ALL());
    const auto first_dst = KE::begin(dst_row);

    using value_type = SourceView::value_type;

    ptrdiff_t result_dist = 0;

    switch (m_api_pick) {
      case 0: {
        auto it =
            KE::inclusive_scan_by_key(member, first_key, last_key, first_src,
                                      first_bool_key_value, first_dst);
        result_dist = KE::distance(first_dst, it);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=, *this] { m_dists_view(row_index) = result_dist; });

        break;
      }

      case 1: {
        auto it     = KE::inclusive_scan_by_key(member, key_row, src_row,
                                                bool_key_value_row, dst_row);
        result_dist = KE::distance(first_dst, it);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=, *this] { m_dists_view(row_index) = result_dist; });

        break;
      }

      case 2: {
        auto it = KE::inclusive_scan_by_key(
            member, first_key, last_key, first_src, first_bool_key_value,
            first_dst, Test::ScanByKey::EqualityFunctor<int>());
        result_dist = KE::distance(first_dst, it);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=, *this] { m_dists_view(row_index) = result_dist; });

        break;
      }

      case 3: {
        auto it = KE::inclusive_scan_by_key(
            member, key_row, src_row, bool_key_value_row, dst_row,
            Test::ScanByKey::EqualityFunctor<bool>());
        result_dist = KE::distance(first_dst, it);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=, *this] { m_dists_view(row_index) = result_dist; });

        break;
      }

      case 4: {
        auto it = KE::inclusive_scan_by_key(
            member, first_key, last_key, first_src, first_bool_key_value,
            first_dst, Test::ScanByKey::EqualityFunctor<bool>(),
            Test::ScanByKey::SumFunctor<value_type>());
        result_dist = KE::distance(first_dst, it);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=, *this] { m_dists_view(row_index) = result_dist; });

        break;
      }

      case 5: {
        auto it = KE::inclusive_scan_by_key(
            member, key_row, src_row, bool_key_value_row, dst_row,
            Test::ScanByKey::EqualityFunctor<int>(),
            Test::ScanByKey::SumFunctor<value_type>());
        result_dist = KE::distance(first_dst, it);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=, *this] { m_dists_view(row_index) = result_dist; });

        break;
      }
    }

    // store result of checking if all members have their local
    // values matching the one stored in m_dists_view
    member.team_barrier();

    const bool intra_team_check = team_members_have_matching_result(
        member, result_dist, m_dists_view(row_index));
    Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
      m_intra_team_sentinel_view(row_index) = intra_team_check;
    });
  }
};

struct InPlace {};

template <class LayoutTag, class ValueType, class KeyType,
          class InPlaceOrVoid = void>
void test_A(std::size_t num_teams, std::size_t num_cols, int api_id) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level inclusive_scan
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  ValueType lower_bound;
  ValueType upper_bound;

  if constexpr (std::is_integral_v<ValueType>) {
    lower_bound = 5;
    upper_bound = 10;
  } else {
    lower_bound = 5;
    upper_bound = 523;
  }

  const auto bounds = make_bounds(lower_bound, upper_bound);

  auto [src_view, src_view_before_op_h] = create_random_view_and_host_clone(
      LayoutTag{}, num_teams, num_cols, bounds, "src_view");

  auto [key_view, key_view_h] =
      Test::ScanByKey::create_2d_key_view_and_host_clone<KeyType>(
          LayoutTag{}, num_teams, num_cols, "key_view");

  Kokkos::View<KE::Impl::KeyValuePair<bool, ValueType>**> bool_key_value_view(
      "bool_key_value_view", num_teams, num_cols);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(num_teams, Kokkos::AUTO());

  // inclusive_scan returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the beginning
  // of the interval that team operates on and then we check that these
  // distances match the std result
  Kokkos::View<std::size_t*> dists_view("dists_view", num_teams);
  // sentinel to check if all members of the team compute the same result
  Kokkos::View<bool*> intra_team_sentinel_view("intra_team_same_result",
                                               num_teams);

  // create the destination view
  Kokkos::View<ValueType**> dst_view("dst_view", num_teams, num_cols);

  if constexpr (std::is_same_v<InPlaceOrVoid, InPlace>) {
    TestFunctorA fnc(key_view, src_view, bool_key_value_view, src_view,
                     dists_view, intra_team_sentinel_view, api_id);
    Kokkos::parallel_for(policy, fnc);
  } else {
    TestFunctorA fnc(key_view, src_view, bool_key_value_view, dst_view,
                     dists_view, intra_team_sentinel_view, api_id);
    Kokkos::parallel_for(policy, fnc);
  }

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto dists_view_h = create_host_space_copy(dists_view);
  auto intra_team_sentinel_view_h =
      create_host_space_copy(intra_team_sentinel_view);
  Kokkos::View<ValueType**, Kokkos::HostSpace> ref_dst_view(
      "ref_dst_view", num_teams, num_cols);

  for (std::size_t i = 0; i < src_view.extent(0); ++i) {
    auto key_row   = Kokkos::subview(key_view_h, i, Kokkos::ALL());
    auto first_key = KE::cbegin(key_row);
    auto last_key  = KE::cend(key_row);
    auto src_row   = Kokkos::subview(src_view_before_op_h, i, Kokkos::ALL());
    auto first_src = KE::cbegin(src_row);
    auto dst_row   = Kokkos::subview(ref_dst_view, i, Kokkos::ALL());
    auto first_dst = KE::begin(dst_row);

    ASSERT_TRUE(intra_team_sentinel_view_h(i));

    switch (api_id) {
      case 0: {
        auto it = Test::ScanByKey::ref_scan_by_key(
            first_key, last_key, first_src, first_dst,
            Test::ScanByKey::EqualityFunctor<bool>(),
            Test::ScanByKey::SumFunctor<value_type>());
        const std::size_t ref_distance = KE::distance(first_dst, it);
        ASSERT_EQ(ref_distance, dists_view_h(i));
        break;
      }
      case 1: {
        auto it = Test::ScanByKey::ref_scan_by_key(
            first_key, last_key, first_src, first_dst,
            Test::ScanByKey::EqualityFunctor<bool>(),
            Test::ScanByKey::SumFunctor<value_type>());
        const std::size_t ref_distance = KE::distance(first_dst, it);
        ASSERT_EQ(ref_distance, dists_view_h(i));
        break;
      }
      case 2: {
        auto it = Test::ScanByKey::ref_scan_by_key(
            first_key, last_key, first_src, first_dst,
            Test::ScanByKey::EqualityFunctor<int>(),
            Test::ScanByKey::SumFunctor<value_type>());
        const std::size_t ref_distance = KE::distance(first_dst, it);
        ASSERT_EQ(ref_distance, dists_view_h(i));
        break;
      }
      case 3: {
        auto it = Test::ScanByKey::ref_scan_by_key(
            first_key, last_key, first_src, first_dst,
            Test::ScanByKey::EqualityFunctor<bool>(),
            Test::ScanByKey::SumFunctor<value_type>());
        const std::size_t ref_distance = KE::distance(first_dst, it);
        ASSERT_EQ(ref_distance, dists_view_h(i));
        break;
      }
      case 4: {
        auto it = Test::ScanByKey::ref_scan_by_key(
            first_key, last_key, first_src, first_dst,
            Test::ScanByKey::EqualityFunctor<bool>(),
            Test::ScanByKey::SumFunctor<value_type>());
        const std::size_t ref_distance = KE::distance(first_dst, it);
        ASSERT_EQ(ref_distance, dists_view_h(i));
        break;
      }
      case 5: {
        auto it = Test::ScanByKey::ref_scan_by_key(
            first_key, last_key, first_src, first_dst,
            Test::ScanByKey::EqualityFunctor<int>(),
            Test::ScanByKey::SumFunctor<value_type>());
        const std::size_t ref_distance = KE::distance(first_dst, it);
        ASSERT_EQ(ref_distance, dists_view_h(i));
        break;
      }
    }
  }

  if constexpr (std::is_same_v<InPlaceOrVoid, InPlace>) {
    auto data_view_after_op_h = create_host_space_copy(src_view);
    expect_equal_host_views(ref_dst_view, data_view_after_op_h);
  } else {
    auto data_view_after_op_h = create_host_space_copy(dst_view);
    expect_equal_host_views(ref_dst_view, data_view_after_op_h);
  }
}

template <class LayoutTag, class ValueType, class InPlaceOrVoid = void>
void run_all_scenarios() {
  for (int num_teams : teamSizesToTest) {
    std::vector<int> num_cols_list;
    // To avoid overflow with multiply based binary operation
    // integral types are tested only with shorter arrays
    if constexpr (std::is_integral_v<ValueType>) {
      num_cols_list = {0, 1, 2, 13, 25};
    } else {
      num_cols_list = {0, 1, 2, 13, 25, 101, 1444, 8153};
    }
    for (const auto& num_cols : num_cols_list) {
      for (int api_id : {0, 1, 2, 3, 4, 5}) {
        if ((api_id == 2) || (api_id == 5))
          test_A<LayoutTag, ValueType, int, InPlaceOrVoid>(num_teams, num_cols,
                                                           api_id);
        else
          test_A<LayoutTag, ValueType, bool, InPlaceOrVoid>(num_teams, num_cols,
                                                            api_id);
      }
    }
  }
}

TEST(std_algorithms_inclusive_scan_by_key_team_test, test) {
// FIXME_OPENMPTARGET
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  GTEST_SKIP() << "the test is known to fail with OpenMPTarget";
#endif
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();

  run_all_scenarios<DynamicTag, double, InPlace>();
  run_all_scenarios<StridedTwoRowsTag, int, InPlace>();
  run_all_scenarios<StridedThreeRowsTag, unsigned, InPlace>();
}

}  // namespace TeamInclusiveScanByKey
}  // namespace stdalgos
}  // namespace Test
