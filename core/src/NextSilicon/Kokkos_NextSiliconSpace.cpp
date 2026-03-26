// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <nextapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

#include <cstdlib>

namespace Kokkos {
namespace Experimental {

// Alignment for allocations in NextSiliconSharedSpace
// Needed as workaround for SW-15953
constexpr size_t SHARED_ALLOC_ALIGNMENT = 65536;  // 2097152

void *NextSiliconSharedSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *NextSiliconSharedSpace::allocate(const char *arg_label,
                                       const size_t arg_alloc_size,
                                       const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *NextSiliconSharedSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  void *ptr = nullptr;

  // NextSilicon implements shared UVM over standard memory operations.
  // Alligned alloc - needed as workaround for SW-15953
  size_t size =
      std::max(align_to_multiple(arg_alloc_size, SHARED_ALLOC_ALIGNMENT),
               SHARED_ALLOC_ALIGNMENT);
  ptr = std::aligned_alloc(SHARED_ALLOC_ALIGNMENT, size);

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void NextSiliconSharedSpace::deallocate(void *const arg_alloc_ptr,
                                        const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void NextSiliconSharedSpace::deallocate(const char *arg_label,
                                        void *const arg_alloc_ptr,
                                        const size_t arg_alloc_size,
                                        const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void NextSiliconSharedSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }

  if (arg_alloc_ptr) {
    // NextSilicon implements shared UVM over standard memory operations.
    free(arg_alloc_ptr);
  }
}

}  // namespace Experimental
}  // namespace Kokkos
