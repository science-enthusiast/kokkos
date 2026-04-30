// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestMDRange.hpp>

namespace Test {

TEST(TEST_CATEGORY, mdrange_2d) {
  TestMDRange_2D<TEST_EXECSPACE>::test_reduce2(100, 100);
  TestMDRange_2D<TEST_EXECSPACE>::test_for2(100, 100);
}

TEST(TEST_CATEGORY, mdrange_array_reduce) {
  TestMDRange_ReduceArray_2D<TEST_EXECSPACE>::test_arrayreduce2(4, 5);
  TestMDRange_ReduceArray_3D<TEST_EXECSPACE>::test_arrayreduce3(4, 5, 10);
}

TEST(TEST_CATEGORY, mdrange_2d_static_batch_size) {
  {
    Test2DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<1>>
        f(1024, 1024);
    f.test_batch_size();
  }
  {
    Test2DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<4>>
        f(1024, 1024);
    f.test_batch_size();
  }

  // md range dim not divisible by static batch size
  {
    Test2DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<8>>
        f(1025, 511);
    f.test_batch_size();
  }

  // md range dim smaller than static batch size
  {
    Test2DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<16>>
        f(3, 2);
    f.test_batch_size();
  }
}

TEST(TEST_CATEGORY, mdrange_5d_static_batch_size) {
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<1>>
        f(16, 16, 16, 16, 16);
    f.test_batch_size();
  }
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<4>>
        f(8, 16, 32, 16, 8);
    f.test_batch_size();
  }

  // md range dim not divisible by static batch size
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<8>>
        f(17, 15, 31, 7, 19);
    f.test_batch_size();
  }

  // md range dim smaller than static batch size
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<16>>
        f(3, 2, 1, 2, 3);
    f.test_batch_size();
  }
}

}  // namespace Test
