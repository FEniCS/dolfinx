# vcpkg overlay port for Intel MPI

This vcpkg overlay port contains scripts for installing Intel MPI on Windows
(only). MSMPI, which is used by default with vcpkg, does not support the MPI3
standard. Using this port requires that Intel OneAPI binaries are already
installed. On Unix systems the built-in OpenMPI or MPICH ports can be used.

From the root of this repository it can be activated by e.g.:

    cmake -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_PORTS="cpp/.vcpkg-overlay" -B build-dir -S cpp/

This overlay port was adapted from the original at:

https://github.com/arcaneframework/framework-ci
