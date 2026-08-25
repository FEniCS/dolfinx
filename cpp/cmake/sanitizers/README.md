# Running DOLFINx under sanitizers

## Configure

```sh
cmake -G Ninja -DCMAKE_BUILD_TYPE=DeveloperDebug -B build -S cpp
```

Sanitizer set: `-DDOLFINX_DEVELOPER_DEBUG_SANITIZERS="address;undefined"`
(default), `=thread`, or `=""` for none.

Suppressions files (`asan.supp`, `lsan.supp`, `ubsan.supp`) live in this
directory and are installed to `${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx/sanitizers`.

## C++

```sh
ASAN_OPTIONS=detect_container_overflow=0:suppressions=$PWD/cpp/cmake/sanitizers/asan.supp \
LSAN_OPTIONS=suppressions=$PWD/cpp/cmake/sanitizers/lsan.supp \
UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PWD/cpp/cmake/sanitizers/ubsan.supp \
ctest --test-dir build --output-on-failure
```

Under MPI, forward the env to the ranks:

```sh
mpiexec -n 3 -x ASAN_OPTIONS -x LSAN_OPTIONS -x UBSAN_OPTIONS ./unittests
```

Notes:
- Don't set `ASAN_OPTIONS=detect_leaks=1` on macOS unless you've confirmed
  your toolchain supports it — it aborts immediately otherwise.
- If MPI/PETSc leak noise dominates, add `detect_leaks=0` and
  `PETSC_OPTIONS=-malloc_debug 0`.

## Python

`dolfinx.cpp` is `dlopen`ed after interpreter start, so the sanitizer
runtime must be preloaded.

**Linux:**

```sh
# GCC: libasan.so   Clang: libclang_rt.asan-x86_64.so
LD_PRELOAD=$($CC -print-file-name=<libname above>) \
ASAN_OPTIONS=detect_container_overflow=0 \
  python -m pytest python/test
```

**macOS:**

```sh
PYBIN="$(python -c 'import sys, pathlib; print(next(pathlib.Path(sys.base_prefix).glob("Resources/Python.app/Contents/MacOS/Python")))')"
DYLD_INSERT_LIBRARIES=$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib) \
ASAN_OPTIONS=detect_container_overflow=0 \
  "$PYBIN" -c "import site; site.addsitedir('$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')')
import sys, pytest; sys.exit(pytest.main(sys.argv[1:]))" python/test
```

Use the venv's actual interpreter binary directly (found above), not the
`python` on `PATH` — otherwise ASan loads twice and aborts with
`Interceptors are not working`.

For MPI, wrap with `mpiexec -x ...` as above.

For a lower-friction check with no preload at all, use
`-DDOLFINX_DEVELOPER_DEBUG_SANITIZERS=undefined`.

## Debugging a run

`-fno-sanitize-recover=all` means the first violation aborts the whole
process, hiding anything later in the run. To see past it, isolate: `ctest
-R <name>` / Catch2 `-r <name>`, or a narrower `pytest` path, one at a time.
