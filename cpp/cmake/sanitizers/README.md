# Running DOLFINx under sanitizers

`-DCMAKE_BUILD_TYPE=DeveloperDebug` prioritises finding bugs over performance:
`-Og -g3`, frame pointers preserved, and AddressSanitizer + UndefinedBehavior­Sanitizer
enabled by default. Configure the sanitizer set with
`-DDOLFINX_DEVELOPER_DEBUG_SANITIZERS="address;undefined"` (the default),
`-DDOLFINX_DEVELOPER_DEBUG_SANITIZERS=thread` (mutually exclusive with
`address`), or `-DDOLFINX_DEVELOPER_DEBUG_SANITIZERS=""` for a plain `-Og`
debug build with no instrumentation.

The suppressions files in this directory (`asan.supp`, `lsan.supp`,
`ubsan.supp`) are installed alongside `DolfinxDeveloperCompilerFlags.cmake`
into `${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx/sanitizers`.

## C++ (`ctest`)

Executables link the sanitizer runtimes directly, so nothing beyond
environment tuning is needed:

```sh
ASAN_OPTIONS=detect_container_overflow=0:suppressions=$PWD/cpp/cmake/sanitizers/asan.supp \
LSAN_OPTIONS=suppressions=$PWD/cpp/cmake/sanitizers/lsan.supp \
UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PWD/cpp/cmake/sanitizers/ubsan.supp \
ctest --test-dir build --output-on-failure
```

`detect_leaks=1` is the default on Linux. On macOS, leak detection is
built into the ASan runtime only on some LLVM/OS combinations; where it is
absent, setting `detect_leaks=1` explicitly aborts immediately with
`AddressSanitizer: detect_leaks is not supported on this platform` rather
than degrading gracefully to "no leak checking" — leave `detect_leaks`
unset on macOS unless you have confirmed your toolchain supports it.
`detect_container_overflow=0` avoids false positives from
`std::vector`/`std::string` objects built inside a non-instrumented
dependency (Catch2, spdlog, ADIOS2, PETSc) and touched by instrumented
DOLFINx code.

For MPI-parallel tests, export the variables into the launched ranks, e.g.
with Open MPI:

```sh
mpiexec -n 3 -x ASAN_OPTIONS -x LSAN_OPTIONS -x UBSAN_OPTIONS ./unittests
```

Open MPI and MPICH both leak by design between `MPI_Init` and
`MPI_Finalize`, and `dlopen` components LSan cannot always symbolise even
with suppressions — `ASAN_OPTIONS=fast_unwind_on_malloc=0` often helps
suppressions match, and `detect_leaks=0` is the pragmatic fallback if MPI
noise dominates a real investigation. With PETSc, pass
`-malloc_debug 0` (or `PETSC_OPTIONS=-malloc_debug 0`) so PETSc's own
allocation tracker does not fight ASan's redzones.

## Python (`pytest`)

`dolfinx.cpp` is a `MODULE` library loaded by `dlopen` after the Python
interpreter has already made allocations, so the sanitizer runtime must be
preloaded into the process from the start.

**Linux, GCC:**

```sh
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
ASAN_OPTIONS=detect_leaks=1:detect_container_overflow=0 \
  python -m pytest python/test
```

**Linux, Clang** (requires the build's `-shared-libasan`, added automatically
when Clang is the compiler on Linux):

```sh
LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so) ...
```

**macOS, Apple Clang:**

```sh
DYLD_INSERT_LIBRARIES=$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib) \
ASAN_OPTIONS=detect_container_overflow=0 \
  python -m pytest python/test
```

**macOS System Integrity Protection caveat:** `DYLD_*` variables are
stripped from the environment of any binary in an SIP-protected location.
`/usr/bin/python3` is protected, and a virtual environment created from it
whose `bin/python` is a *symlink* inherits the protection. Use a Homebrew or
conda interpreter, or create the venv with `python -m venv --copies`.
Symptom if this is wrong: `Interceptors are not working. This may be because
AddressSanitizer is loaded too late`.

**Lower-friction entry point:** `-DDOLFINX_DEVELOPER_DEBUG_SANITIZERS=undefined`
needs no preload at all, since UBSan has no allocator to install early. Start
there when investigating a Python-side issue before reaching for ASan.

MPI-parallel Python tests need the same `mpiexec -x` forwarding as the C++
case above.

## Known findings

`DeveloperDebug` is opt-in and not gated by CI, so the test suite is not
expected to pass cleanly from day one — see the tracking issue for the
current list. `-fno-sanitize-recover=all` means the first hit aborts the
whole process (Catch2's `CHECK_NOTHROW` cannot catch a `SIGABRT`), so a
single finding can mask others further into a test run; re-run with
`-r <failing test name>` to isolate one at a time. Do not weaken the
build-time flags to make a run look green — use the run-time options above
instead.
