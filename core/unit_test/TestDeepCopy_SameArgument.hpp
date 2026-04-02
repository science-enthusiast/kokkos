// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

#define KOKKOS_TEST_EXPECT_WARNING(CODE, SUBSTRING)                       \
  do {                                                                    \
    ::testing::internal::CaptureStderr();                                 \
    CODE;                                                                 \
    auto captured = ::testing::internal::GetCapturedStderr();             \
    if (Kokkos::show_warnings()) {                                        \
      EXPECT_FALSE(captured.empty());                                     \
      EXPECT_NE(captured.find(SUBSTRING), std::string::npos) << captured; \
    } else {                                                              \
      EXPECT_TRUE(captured.empty()) << captured;                          \
    }                                                                     \
  } while (false)

template <class View>
void test_deep_copy_same_argument(View v) {
  auto v_h = Kokkos::create_mirror_view(v);
  EXPECT_NO_THROW(Kokkos::deep_copy(v, v_h));
  EXPECT_NO_THROW(Kokkos::deep_copy(v_h, v));
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
  KOKKOS_TEST_EXPECT_WARNING(Kokkos::deep_copy(v, v),
                             "Kokkos WARNING: deep_copy() source and "
                             "destination View arguments are identical");

  KOKKOS_TEST_EXPECT_WARNING(Kokkos::deep_copy(v_h, v_h),
                             "Kokkos WARNING: deep_copy() source and "
                             "destination View arguments are identical");

#else
  EXPECT_DEATH(Kokkos::deep_copy(v, v),
               "Kokkos ERROR: deep_copy\\(\\) source and "
               "destination View arguments are identical");
  EXPECT_DEATH(Kokkos::deep_copy(v_h, v_h),
               "Kokkos ERROR: deep_copy\\(\\) source and "
               "destination View arguments are identical");
#endif

  TEST_EXECSPACE exec;
  EXPECT_NO_THROW(Kokkos::deep_copy(exec, v, v_h));
  EXPECT_NO_THROW(Kokkos::deep_copy(exec, v_h, v));
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
  KOKKOS_TEST_EXPECT_WARNING(Kokkos::deep_copy(exec, v, v),
                             "Kokkos WARNING: deep_copy() source and "
                             "destination View arguments are identical");

  KOKKOS_TEST_EXPECT_WARNING(Kokkos::deep_copy(exec, v_h, v_h),
                             "Kokkos WARNING: deep_copy() source and "
                             "destination View arguments are identical");

#else
  EXPECT_DEATH(Kokkos::deep_copy(exec, v, v),
               "Kokkos ERROR: deep_copy\\(\\) source and "
               "destination View arguments are identical");
  EXPECT_DEATH(Kokkos::deep_copy(exec, v_h, v_h),
               "Kokkos ERROR: deep_copy\\(\\) source and "
               "destination View arguments are identical");
#endif
}

TEST(TEST_CATEGORY_DEATH, deep_copy_same_view) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  test_deep_copy_same_argument(Kokkos::View<double, TEST_EXECSPACE>("v0"));
  test_deep_copy_same_argument(Kokkos::View<int*, TEST_EXECSPACE>("v1", 10));
}

}  // namespace
