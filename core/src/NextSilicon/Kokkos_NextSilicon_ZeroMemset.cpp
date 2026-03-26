// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <nextapi/intrinsics.h>
#include <nextapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ZeroMemset.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>

#ifdef KOKKOS_ENABLE_NEXTSILICON

namespace Kokkos {
namespace Impl {
void ZeroMemsetNextSilicon(void* buffer, size_t buffer_size) {
  uint8_t pattern = 0;
  nextapi_memory_fill(buffer, &pattern, sizeof(pattern), buffer_size);
}

}  // namespace Impl
}  // namespace Kokkos

#endif
