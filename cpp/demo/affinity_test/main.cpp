// ```text
// Copyright (C) 2026 Jack S. Hale
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Thread affinity and memory bandwidth
//
// This demo measures the effect of pinning the worker threads used by
// `mesh::compute_entities` to individual cores, via the
// `mesh::AffinityPolicy` callback. `compute_entities`'s threaded phase
// (`build_entity_list`) is memory-bandwidth bound: each thread streams
// through a slice of the cell-to-vertex array and writes into a large
// shared output array, with very little arithmetic per byte moved. For
// a kernel like this, keeping each thread's memory traffic local to
// the NUMA domain the calling MPI rank is already bound to (e.g. via
// `mpirun --bind-to numa`) should measurably increase achieved
// bandwidth over letting the OS scheduler place threads freely.
//
// On Linux, pinning is a real, kernel-enforced affinity mask set with
// `pthread_setaffinity_np`. On macOS there is no equivalent kernel
// primitive -- `thread_policy_set`'s `THREAD_AFFINITY_POLICY` is only a
// same-L2-group scheduling hint and is not honoured on Apple Silicon --
// so the "pinned" arm there falls back to a QoS hint and should be
// expected to show little or no difference from the unpinned arm.

#include <chrono>
#include <cstdint>
#include <dolfinx.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <format>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

using namespace dolfinx;

namespace
{
/// @brief Number of logical CPUs available to the calling process.
///
/// On Linux this is the size of the process's cpuset, i.e. what an MPI
/// launcher's `--bind-to numa`/`--bind-to core` already restricted this
/// rank to. There is no cpuset concept on macOS, so
/// `std::thread::hardware_concurrency()` is used there instead.
int available_cores()
{
#if defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
  {
    if (int n = CPU_COUNT(&mask); n > 0)
      return n;
  }
#endif
  return static_cast<int>(std::thread::hardware_concurrency());
}

/// @brief Build an mesh::AffinityPolicy that pins worker thread `i` to
/// the `i`-th core in the calling process's cpuset (round-robin if
/// there are more threads than cores).
///
/// Runs on the worker thread itself, so it pins the thread's own
/// native handle.
mesh::AffinityPolicy make_pin_to_core_policy()
{
#if defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  sched_getaffinity(0, sizeof(mask), &mask);
  std::vector<int> cores;
  for (int c = 0; c < CPU_SETSIZE; ++c)
    if (CPU_ISSET(c, &mask))
      cores.push_back(c);

  return [cores](int i, int /*num_threads*/)
  {
    if (cores.empty())
      return;
    cpu_set_t m;
    CPU_ZERO(&m);
    CPU_SET(cores[i % cores.size()], &m);
    pthread_setaffinity_np(pthread_self(), sizeof(m), &m);
  };
#elif defined(__APPLE__)
  return [](int /*i*/, int /*num_threads*/)
  {
    // Best-effort scheduling hint only -- macOS has no per-core pinning
    // primitive.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  };
#else
  return nullptr;
#endif
}

/// @brief Time `compute_entities(dim=1)` on a fresh hexahedral box mesh
/// built on `MPI_COMM_SELF`, for the given number of threads and
/// affinity policy. Returns (seconds, achieved GB/s), where the byte
/// count is a lower-bound estimate covering only the two output arrays
/// written by `build_entity_list`.
template <std::floating_point T>
std::pair<double, double> bench(std::int64_t n, int num_threads,
                                const mesh::AffinityPolicy& policy)
{
  mesh::Mesh<T> mesh
      = mesh::create_box<T>(MPI_COMM_SELF, {{{0, 0, 0}, {1, 1, 1}}}, {n, n, n},
                            mesh::CellType::hexahedron);
  auto topology = mesh.topology_mutable();

  auto t0 = std::chrono::steady_clock::now();
  auto [c_to_e, e_to_v, im, ghosts] = mesh::compute_entities(
      *topology, 1, mesh::CellType::interval, num_threads, policy);
  auto t1 = std::chrono::steady_clock::now();

  double seconds = std::chrono::duration<double>(t1 - t0).count();
  std::size_t num_edges = e_to_v->num_nodes();
  std::size_t bytes = 2 * 2 * num_edges * sizeof(std::int32_t);
  double gbps = bytes / seconds / 1e9;
  return {seconds, gbps};
}

/// @brief Run `bench` `repeats` times and return the minimum time (and
/// corresponding bandwidth), the standard benchmarking convention for
/// suppressing scheduler/OS noise rather than averaging it in.
template <std::floating_point T>
std::pair<double, double> bench_min(std::int64_t n, int num_threads,
                                    const mesh::AffinityPolicy& policy,
                                    int repeats)
{
  auto best = bench<T>(n, num_threads, policy);
  for (int r = 1; r < repeats; ++r)
  {
    auto result = bench<T>(n, num_threads, policy);
    if (result.first < best.first)
      best = result;
  }
  return best;
}

template <std::floating_point T>
void run(std::int64_t n, int max_threads, int repeats)
{
  mesh::AffinityPolicy pin = make_pin_to_core_policy();

  std::cout << std::format("{:>8} {:>12} {:>10} {:>12} {:>10} {:>9}\n",
                           "threads", "unpinned", "GB/s", "pinned", "GB/s",
                           "speedup");
  for (int t = 1; t <= max_threads; ++t)
  {
    auto [t_unpinned, bw_unpinned] = bench_min<T>(n, t, nullptr, repeats);
    auto [t_pinned, bw_pinned] = bench_min<T>(n, t, pin, repeats);
    std::cout << std::format(
        "{:>8} {:>10.4f}s {:>10.2f} {:>10.4f}s {:>10.2f} {:>8.2f}x\n", t,
        t_unpinned, bw_unpinned, t_pinned, bw_pinned, t_unpinned / t_pinned);
    std::cout.flush();
  }
}
} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);
  {
    int max_threads = available_cores();
    std::cout << "Detected " << max_threads
              << " logical CPU(s) available to this process\n";
#if defined(__linux__)
    std::cout << "Platform: Linux -- pinning via pthread_setaffinity_np "
                 "is kernel-enforced\n";
#elif defined(__APPLE__)
    std::cout << "Platform: macOS -- no kernel core-pinning primitive "
                 "exists; the \"pinned\" column is a QoS hint only and "
                 "may show little or no difference from \"unpinned\"\n";
#endif

    // Number of cells per direction for the box mesh. 96^3 hexahedra
    // gives ~2.4M cells / ~14M edges, large enough that the working set
    // exceeds cache and the kernel is genuinely bandwidth-bound.
    constexpr std::int64_t n = 96;
    constexpr int repeats = 3;
    run<double>(n, max_threads, repeats);
  }
  MPI_Finalize();
  return 0;
}
