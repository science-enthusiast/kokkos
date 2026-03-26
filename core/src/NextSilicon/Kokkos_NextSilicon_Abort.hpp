// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_ABORT_HPP
#define KOKKOS_NEXTSILICON_ABORT_HPP

namespace Kokkos {
namespace Impl {

[[noreturn]] inline void nextsilicon_abort(char const* msg) {
  // FIXME_NEXTSILICON Add nextapi to abort application from device
  while (true)
    ;
}

}  // namespace Impl
}  // namespace Kokkos

#endif
