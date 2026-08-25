# Running DOLFINx under sanitizers

## Build the C++ library

```sh
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS_DEBUG="" -DCMAKE_C_FLAGS_DEBUG="" \
  -DCMAKE_CXX_FLAGS="-Og -g3 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-sanitize=vptr,function -fno-omit-frame-pointer -D_GLIBCXX_ASSERTIONS -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG" \
  -DCMAKE_C_FLAGS="-Og -g3 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DBUILD_TESTING=ON -DDOLFINX_BUILD_DEMOS=ON -B build -S cpp
cmake --build build
cmake --install build
```

`-O0` rather than `-Og` may be best for step-through debugging sessions.

For UBSan only (required for MPI-parallel Python on macOS, see below): drop
`address,` from both `-fsanitize=` values and drop the two `LINKER_FLAGS`
lines.

## Run under C++

```sh
ASAN_OPTIONS=detect_container_overflow=0:suppressions=$PWD/cpp/cmake/sanitizers/asan.supp \
LSAN_OPTIONS=suppressions=$PWD/cpp/cmake/sanitizers/lsan.supp \
UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PWD/cpp/cmake/sanitizers/ubsan.supp \
ctest --test-dir build --output-on-failure
```

`ASAN_OPTIONS=detect_leaks=1` aborts immediately on macOS unless your
toolchain supports it — leave it unset there.

Under MPI:

```sh
export ASAN_OPTIONS=detect_container_overflow=0:suppressions=$PWD/cpp/cmake/sanitizers/asan.supp
export LSAN_OPTIONS=suppressions=$PWD/cpp/cmake/sanitizers/lsan.supp
export UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PWD/cpp/cmake/sanitizers/ubsan.supp
mpiexec -n 3 -x ASAN_OPTIONS -x LSAN_OPTIONS -x UBSAN_OPTIONS ./build/test/unittests
```

## Build the Python bindings

```sh
pip install --check-build-dependencies --no-build-isolation \
  -Ccmake.define.CMAKE_BUILD_TYPE="Debug" \
  -Ccmake.define.CMAKE_CXX_FLAGS_DEBUG="" \
  -Ccmake.define.CMAKE_CXX_FLAGS="-Og -g3 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-sanitize=vptr,function -fno-omit-frame-pointer -D_GLIBCXX_ASSERTIONS -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG" \
  python/
```

## Run under Python

**Linux, GCC:**

```sh
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
ASAN_OPTIONS=detect_container_overflow=0 \
  python -m pytest python/test
```

Under MPI:

```sh
export LD_PRELOAD=$(gcc -print-file-name=libasan.so)
export ASAN_OPTIONS=detect_container_overflow=0
mpiexec -n 3 -x LD_PRELOAD -x ASAN_OPTIONS python -m pytest python/test
```

**Linux, Clang:**

```sh
LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-$(uname -m).so) \
ASAN_OPTIONS=detect_container_overflow=0 \
  python -m pytest python/test
```

Under MPI:

```sh
export LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-$(uname -m).so)
export ASAN_OPTIONS=detect_container_overflow=0
mpiexec -n 3 -x LD_PRELOAD -x ASAN_OPTIONS python -m pytest python/test
```

**macOS:** plain `python -m pytest` with `DYLD_INSERT_LIBRARIES` set aborts
with `Interceptors are not working` (the venv's `python` re-launches itself
through `Resources/Python.app/...`, double-loading ASan) — launch the inner
binary directly instead:

```sh
cat > /tmp/run_pytest_sanitized.py <<'EOF'
import os, site, sys
site.addsitedir(os.path.join(os.environ["VIRTUAL_ENV"], "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"))
import pytest
sys.exit(pytest.main(sys.argv[1:]))
EOF
```

```sh
PYBIN="$(python -c 'import sys, pathlib; print(next(pathlib.Path(sys.base_prefix).glob("Resources/Python.app/Contents/MacOS/Python")))')"
DYLD_INSERT_LIBRARIES=$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib) \
ASAN_OPTIONS=detect_container_overflow=0 \
  "$PYBIN" /tmp/run_pytest_sanitized.py python/test
```

Under MPI (UBSan-only build, per above — Open MPI strips
`DYLD_INSERT_LIBRARIES` before launching each rank):

```sh
export UBSAN_OPTIONS=print_stacktrace=1
mpiexec -n 3 -x UBSAN_OPTIONS python -m pytest python/test
```
