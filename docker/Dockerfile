# Dockerfile describing development builds of FEniCSx
#
# Authors:
# Jack S. Hale <jack.hale@uni.lu>
# Lizao Li <lzlarryli@gmail.com>
# Garth N. Wells <gnw20@cam.ac.uk>
# Jan Blechta <blechta@karlin.mff.cuni.cz>
#
# To run a nightly build:
#
#    docker run -ti dolfinx/dolfinx
#
# To run a Jupyter lab session:
#
#    docker run --init -p 8888:8888 dolfinx/lab
#
# To run and share the current host directory with the container:
#
#    docker run --init -p 8888:8888 -v "$(pwd)":/root/shared dolfinx/lab
#
# To build from source, first checkout the DOLFINx, FFCx, Basix and UFL
# repositories into the working directory, e.g.:
#
# $ ls $(pwd)
# dolfinx  ffcx  basix  ufl
#
# Then run the commands:
#
#    docker pull dolfinx/dolfinx-onbuild
#    echo "FROM dolfinx/dolfinx-onbuild" | docker build -f- .
#
# You can build an optimised version of the complete FEniCS environment for
# your platform using the commands:
#
#    echo '{ "cffi_extra_compile_args" : ["-O2", "-march=native" ] }' > dolfinx/docker/dolfinx_jit_parameters.json
#    docker build --target dolfinx --file dolfinx/docker/Dockerfile --build-arg PETSC_SLEPC_OPTFLAGS="-O2 -march=native" --build-arg DOLFINX_CMAKE_CXX_FLAGS="-march=native" .
#
# You can build an optimised version of the FEniCS development environment
# (without the FEniCS components) for your platform using the command:
#
#    docker build --target dev-env --file dolfinx/docker/Dockerfile --build-arg PETSC_SLEPC_OPTFLAGS="-O2 -march=native" .
#

ARG GMSH_VERSION=4.6.0
ARG PYBIND11_VERSION=2.6.2
ARG PETSC_VERSION=3.15.0
ARG SLEPC_VERSION=3.15.0
ARG SLEPC4PY_VERSION=3.15.0
ARG ADIOS2_VERSION=2.7.1
# TODO: Update KaHIP version upon a release.
ARG KAHIP_VERSION=623decb
ARG XTENSOR_VERSION=0.23.9
ARG XTENSOR_BLAS_VERSION=0.19.1
ARG XTL_VERSION=0.7.2

########################################

FROM ubuntu:20.04 as dev-env
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"
LABEL description="FEniCS testing and development environment with PETSc real, complex, 32-bit and 64-bit modes"

ARG GMSH_VERSION
ARG PYBIND11_VERSION
ARG PETSC_VERSION
ARG SLEPC_VERSION
ARG SLEPC4PY_VERSION
ARG ADIOS2_VERSION
ARG KAHIP_VERSION
ARG XTENSOR_VERSION
ARG XTENSOR_BLAS_VERSION
ARG XTL_VERSION

# The following ARGS are used in the dev-env layer.
# They are safe defaults. They can be overridden by the user.
# Compiler optimisation flags for SLEPc and PETSc, all languages.
ARG PETSC_SLEPC_OPTFLAGS="-O2"
# PETSc and SLEPc number of make processes (--with-make-np)
ARG PETSC_SLEPC_MAKE_NP=2
# Turn on PETSc and SLEPc debugging. "yes" or "no".
ARG PETSC_SLEPC_DEBUGGING="no"
# Ubuntu MPI variant. "mpich" or "openmpi".
ARG MPI="mpich"

WORKDIR /tmp

# Environment variables
ENV OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0

# Install dependencies available via apt-get.
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or tests.
# - Third set of packages are optional, but required to run gmsh
#   pre-built binaries.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    clang-10 \
    cmake \
    g++ \
    gfortran \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-timer-dev \
    libhdf5-${MPI}-dev \
    liblapack-dev \
    lib${MPI}-dev \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-setuptools && \
    #
    apt-get -y install \
    doxygen \
    git \
    graphviz \
    valgrind \
    wget && \
    #
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxft2 \
    libxinerama1 && \
    #
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install Python packages (via pip)
# - First set of packages are required to build and run DOLFINx Python.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
# LLVM_CONFIG required on aarch64, should be removed long-term.
RUN LLVM_CONFIG=/usr/bin/llvm-config-10 pip3 install --no-cache-dir mpi4py numba && \
    pip3 install --no-cache-dir cffi cppimport flake8 pytest pytest-xdist sphinx sphinx_rtd_theme

# Upgrade numpy via pip. Exclude binaries to avoid conflicts with libblas
# (See issue #126 and #1305)
RUN pip3 install --no-binary="numpy" numpy --upgrade

# Install xtl, xtensor, xtensor-blas.
RUN git clone -b ${XTL_VERSION} --single-branch https://github.com/xtensor-stack/xtl.git && \
    cd xtl && \
    cmake -G Ninja . && \
    ninja install && \
    cd ../ && \
    git clone -b ${XTENSOR_VERSION} --single-branch https://github.com/xtensor-stack/xtensor.git && \
    cd xtensor && \
    cmake -G Ninja . && \
    ninja install && \
    cd ../ && \
    git clone -b ${XTENSOR_BLAS_VERSION} --single-branch https://github.com/xtensor-stack/xtensor-blas.git && \
    cd xtensor-blas && \
    cmake -G Ninja . && \
    ninja install && \
    rm -rf xtl xtensor xtensor-blas

# Install pybind11
RUN wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make install && \
    rm -rf /tmp/*

# Install KaHIP
ENV KAHIP_DIR=/usr/local/KaHIP/deploy
RUN cd /usr/local && \
    git clone https://github.com/schulzchristian/KaHIP.git && \
    cd KaHIP/ && \
    git checkout ${KAHIP_VERSION} && \
    mkdir build && cd build && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=${KAHIP_DIR} -DNONATIVEOPTIMIZATIONS=On .. && \
    ninja install

# Install ADIOS2
RUN wget -nc --quiet https://github.com/ornladios/ADIOS2/archive/v${ADIOS2_VERSION}.tar.gz -O adios2-v${ADIOS2_VERSION}.tar.gz && \
    mkdir -p adios2-v${ADIOS2_VERSION} && \
    tar -xf adios2-v${ADIOS2_VERSION}.tar.gz -C adios2-v${ADIOS2_VERSION} --strip-components 1 && \
    cmake -DADIOS2_USE_Fortran=Off -DBUILD_TESTING=Off -DADIOS2_BUILD_EXAMPLES=Off -DADIOS2_USE_ZeroMQ=Off  -B build-dir -S ./adios2-v${ADIOS2_VERSION} && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf /tmp/*

# Download Install gmsh SDK
# Only compatible with x86-64 (amd64)
RUN dpkgArch="$(dpkg --print-architecture)"; \
    case "$dpkgArch" in amd64) \
      cd /usr/local && \
      wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
      tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
      rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz ;; \
    esac;

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH \
    PYTHONPATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/lib:$PYTHONPATH

# Install PETSc and petsc4py with real and complex types
ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p ${PETSC_DIR} && tar -xf petsc-${PETSC_VERSION}.tar.gz -C ${PETSC_DIR} --strip-components 1 && \
    cd ${PETSC_DIR} && \
    # Real, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-real-32 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=real && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-real-32 ${MAKEFLAGS} all && \
    # Complex, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-complex-32 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=complex && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-complex-32 ${MAKEFLAGS} all && \
    # Real, 64-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-real-64 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=yes \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu_dist \
    --with-scalar-type=real && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-real-64 ${MAKEFLAGS} all && \
    # Complex, 64-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-complex-64 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=yes \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu_dist \
    --with-scalar-type=complex && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-complex-64 ${MAKEFLAGS} all && \
    # Install petsc4py
    cd src/binding/petsc4py && \
    PETSC_ARCH=linux-gnu-real-32:linux-gnu-complex-32:linux-gnu-real-64:linux-gnu-complex-64 pip3 install --no-cache-dir . && \
    # Cleanup
    apt-get -y purge bison flex && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf \
    ${PETSC_DIR}/**/tests/ \
    ${PETSC_DIR}/**/obj/ \
    ${PETSC_DIR}/**/externalpackages/  \
    ${PETSC_DIR}/CTAGS \
    ${PETSC_DIR}/RDict.log \
    ${PETSC_DIR}/TAGS \
    ${PETSC_DIR}/docs/ \
    ${PETSC_DIR}/share/ \
    ${PETSC_DIR}/src/ \
    ${PETSC_DIR}/systems/ \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install SLEPc
RUN wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p ${SLEPC_DIR} && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C ${SLEPC_DIR} --strip-components 1 && \
    cd ${SLEPC_DIR} && \
    export PETSC_ARCH=linux-gnu-real-32 && \
    python3 ./configure && \
    make && \
    export PETSC_ARCH=linux-gnu-complex-32 && \
    python3 ./configure && \
    make && \
    export PETSC_ARCH=linux-gnu-real-64 && \
    python3 ./configure && \
    make && \
    export PETSC_ARCH=linux-gnu-complex-64 && \
    python3 ./configure && \
    make && \
    # Install slepc4py
    cd src/binding/slepc4py && \
    PETSC_ARCH=linux-gnu-real-32:linux-gnu-complex-32:linux-gnu-real-64:linux-gnu-complex-64 pip3 install --no-cache-dir . && \
    rm -rf ${SLEPC_DIR}/CTAGS ${SLEPC_DIR}/TAGS ${SLEPC_DIR}/docs ${SLEPC_DIR}/src/ ${SLEPC_DIR}/**/obj/ ${SLEPC_DIR}/**/test/ && \
    rm -rf /tmp/*

WORKDIR /root

########################################

FROM dev-env as dolfinx-onbuild
LABEL description="DOLFINx in 32-bit real and complex modes (onbuild)"

ADD dolfinx/docker/dolfinx-real-mode /usr/local/bin/dolfinx-real-mode
ADD dolfinx/docker/dolfinx-complex-mode /usr/local/bin/dolfinx-complex-mode
RUN chmod +x /usr/local/bin/dolfinx-*-mode

ONBUILD WORKDIR /src

# This leaves the sources inside the container. This is a limitation of Docker.
# There is some trickery in the intermediate and DOLFINx containers that can be
# used to remove this source if needed, see below.
ONBUILD ADD basix/ /src/basix/
ONBUILD ADD ufl/ /src/ufl/
ONBUILD ADD ffcx/ /src/ffcx/
ONBUILD ADD dolfinx/ /src/dolfinx/

# These files are empty by default, i.e. they do nothing.
# The user can set them at build time if they wish.
ONBUILD ADD dolfinx/docker/dolfinx_jit_parameters.json /root/.config/dolfinx/dolfinx_jit_parameters.json
ONBUILD ADD dolfinx/docker/ffcx_parameters.json /root/.config/ffcx/ffcx_parameters.json

# The following ARGS are used in the DOLFINx layer.
# They are safe defaults.
# CMake build type for DOLFINx C++ build. See CMake documentation.
ONBUILD ARG DOLFINX_CMAKE_BUILD_TYPE="RelWithDebInfo"
# Extra CMake C++ compiler flags for DOLFINx C++ build.
ONBUILD ARG DOLFINX_CMAKE_CXX_FLAGS

# The dolfinx-onbuild container expects to have folders basix/ ufl/ ffcx/ and
# dolfinx/ mounted/shared at /src.
ONBUILD RUN cd basix && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFINX_CMAKE_CXX_FLAGS} -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    python3 -m pip install ./python && \
    cd ../ufl && pip3 install --no-cache-dir . && \
    cd ../ffcx && pip3 install --no-cache-dir . && \
    cd ../ && pip3 install --no-cache-dir ipython

ONBUILD RUN cd dolfinx && \
    mkdir -p build-real && \
    cd build-real && \
    PETSC_ARCH=linux-gnu-real-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFINX_CMAKE_CXX_FLAGS} ../cpp && \
    ninja install && \
    cd ../python && \
    CXXFLAGS=${DOLFINX_CMAKE_CXX_FLAGS} PETSC_ARCH=linux-gnu-real-32 pip3 install --target /usr/local/dolfinx-real/lib/python3.8/dist-packages --no-dependencies . && \
    cd ../ && \
    mkdir -p build-complex && \
    cd build-complex && \
    PETSC_ARCH=linux-gnu-complex-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFIN_CMAKE_CXX_FLAGS} ../cpp && \
    ninja install && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cd ../python && \
    CXXFLAGS=${DOLFINX_CMAKE_CXX_FLAGS} PETSC_ARCH=linux-gnu-complex-32 pip3 install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies .

# Real by default.
ONBUILD ENV PKG_CONFIG_PATH=/usr/local/dolfinx-real/lib/pkgconfig:$PKG_CONFIG_PATH \
    PETSC_ARCH=linux-gnu-real-32 \
    PYTHONPATH=/usr/local/dolfinx-real/lib/python3.8/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-real/lib:$LD_LIBRARY_PATH

ONBUILD WORKDIR /root

########################################

FROM dolfinx-onbuild as intermediate

########################################

FROM dev-env as dolfinx
LABEL description="DOLFINx in 32-bit real and complex modes"

# This layer manually copies the build artifacts from intermediate into dev-env
# to make the final image. This is a workaround for a well known limitation of
# Docker that you cannot cleanup after an ADD operation. This reduces the
# container size by around 80MB as the /src folder no longer exists in the final
# image.
COPY --from=intermediate /usr/local /usr/local
COPY --from=intermediate /root/.config /root/.config

# Real by default.
# Note that because we inherit from dev-env we do not inherit these ENV from
# dolfinx-onbuild so this must be repeated here.
ENV PKG_CONFIG_PATH=/usr/local/dolfinx-real/lib/pkgconfig:$PKG_CONFIG_PATH \
    PETSC_ARCH=linux-gnu-real-32 \
    PYTHONPATH=/usr/local/dolfinx-real/lib/python3.8/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-real/lib:$LD_LIBRARY_PATH

########################################

FROM dolfinx as lab
LABEL description="DOLFINx Jupyter Lab"

WORKDIR /root

RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# pyvista dependencies from pip. Only compatible with x86-64 (amd64).
# matplotlib improves plotting quality with better color maps and properly rendering colorbars.
RUN dpkgArch="$(dpkg --print-architecture)"; \
    case "$dpkgArch" in amd64) \
      pip3 install --no-cache-dir pyvista ;; \
    esac; \
    pip3 install --no-cache-dir matplotlib

EXPOSE 8888/tcp
ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
