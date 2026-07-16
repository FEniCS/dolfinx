# vcpkg overlay ports

## Intel MPI

This vcpkg overlay port contains scripts for installing Intel MPI on Windows
(only). MSMPI, which is used by default with vcpkg, does not support the MPI3
standard. Using this port requires that Intel OneAPI binaries are already
installed. On Unix systems the built-in OpenMPI or MPICH ports can be used.

This overlay port was adapted from the original at:

https://github.com/arcaneframework/framework-ci

## libaec

The upstream vcpkg port for `libaec` downloads from `https://gitlab.dkrz.de`,
which can intermittently fail with HTTP 429 (rate limiting) errors in CI.
This overlay port redirects the download to the GitHub mirror at
https://github.com/MathisRosenhauer/libaec.

## Usage

From the root of this repository it can be activated by e.g.:

    cmake -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_PORTS="cpp/.vcpkg-overlay" -B build-dir -S cpp/
