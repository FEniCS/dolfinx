# Dockerfile describing development builds of FEniCS-X
#
# Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>
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
# To build from source, first checkout the DOLFINX, FFCX, FIAT and UFL
# repositories into the working directory, e.g.:
#
# $ ls $(pwd)
# dolfinx  ffcx  fiat  ufl
#
# Then run the commands:
#
#    docker pull dolfinx/dolfinx-onbuild
#    echo "FROM dolfinx/dolfinx-onbuild" | docker build -f- . 
#
# You can build an optimised version of the complete FEniCS environment for
# your platform using the command:
#
#    docker build --target dolfinx --file dolfinx/docker/Dockerfile --build-arg PETSC_SLEPC_OPTFLAGS="-O2 -march=native" --build-arg DOLFINX_CMAKE_CXX_FLAGS="-march=native" .
#
# The pre-built images on Dockerhub are currently optimised for Intel Sandy
# Bridge architectures and will run on all modern x86-64 chips (>~2012).
#
# You can build an optimised version of the FEniCS development environment
# (without the FEniCS components) for your platform using the command:
#
#    docker build --target dev-env --file dolfinx/docker/Dockerfile --build-arg PETSC_SLEPC_OPTFLAGS="-O2 -march=native" .
#

ARG GMSH_VERSION=4.6.0
ARG PYBIND11_VERSION=2.5.0
ARG PETSC_VERSION=3.13.3
ARG SLEPC_VERSION=3.13.3
ARG PETSC4PY_VERSION=3.13.0
ARG SLEPC4PY_VERSION=3.13.0
# Should be updated upon a new KaHIP release
ARG KAHIP_VERSION=14be06c

# Flags to pass to ninja build system.
ARG MAKEFLAGS
# Compiler optimisation flags for SLEPc and PETSc, all languages.
ARG PETSC_SLEPC_OPTFLAGS="-O2 -march=sandybridge"
# Turn on PETSc and SLEPc debugging. "yes" or "no".
ARG PETSC_SLEPC_DEBUGGING="no"
# CMake build type for DOLFINX C++ build. See CMake documentation.
ARG DOLFINX_CMAKE_BUILD_TYPE="RelWithDebInfo"
# Extra CMake C++ compiler flags for DOLFINX C++ build.
ARG DOLFINX_CMAKE_CXX_FLAGS="-march=sandybridge"
# Ubuntu MPI variant. "mpich" or "openmpi".
ARG MPI="mpich"

########################################

FROM ubuntu:20.04 as dev-env
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"
LABEL description="FEniCS testing and development environment with PETSc real, complex, 32-bit and 64-bit modes"

ARG GMSH_VERSION
ARG PYBIND11_VERSION
ARG PETSC_VERSION
ARG PETSC4PY_VERSION
ARG SLEPC_VERSION
ARG SLEPC4PY_VERSION

ARG MAKEFLAGS
ARG PETSC_SLEPC_OPTFLAGS
ARG PETSC_SLEPC_DEBUGGING
ARG MPI

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
    libboost-iostreams-dev \
    libboost-math-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-timer-dev \
    libeigen3-dev \
    libhdf5-${MPI}-dev \
    liblapack-dev \
    lib${MPI}-dev \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-setuptools && \
    #
    apt-get -y install \
    doxygen \
    git \
    graphviz \
    sudo \
    valgrind \
    wget && \
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxinerama1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install Python packages (via pip)
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
RUN pip3 install --no-cache-dir mpi4py numba && \
    pip3 install --no-cache-dir cffi cppimport flake8 pytest pytest-xdist sphinx sphinx_rtd_theme

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
RUN cd /usr/local && \
    git clone https://github.com/schulzchristian/KaHIP.git && \
    cd KaHIP/ && \
    git checkout $KAHIP_VERSION && \
    ./compile_withcmake.sh

ENV KAHIP_ROOT=/usr/local/KaHIP

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

# Add gmsh python API
ENV PYTHONPATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/lib:$PYTHONPATH

# Install PETSc with real and complex types
ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc
WORKDIR /tmp
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p ${PETSC_DIR} && tar -xf petsc-${PETSC_VERSION}.tar.gz -C ${PETSC_DIR} --strip-components 1 && \
    cd ${PETSC_DIR} && \
    # Real, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-real-32 \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-blacs \
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
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-blacs \
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
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
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
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
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
WORKDIR /tmp
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
    rm -rf ${SLEPC_DIR}/CTAGS ${SLEPC_DIR}/TAGS ${SLEPC_DIR}/docs ${SLEPC_DIR}/src/ ${SLEPC_DIR}/**/obj/ ${SLEPC_DIR}/**/test/ && \
    rm -rf /tmp/*

# Install petsc4py and slepc4py with real and complex types
RUN PETSC_ARCH=linux-gnu-real-32:linux-gnu-complex-32:linux-gnu-real-64:linux-gnu-complex-64 pip3 install --no-cache-dir petsc4py==${PETSC4PY_VERSION} slepc4py==${SLEPC4PY_VERSION}

WORKDIR /root

########################################

FROM dev-env as dolfinx-onbuild
LABEL description="DOLFIN-X in 32-bit real and complex modes (onbuild)"

ADD dolfinx/docker/dolfinx-real-mode /usr/local/bin/dolfinx-real-mode
ADD dolfinx/docker/dolfinx-complex-mode /usr/local/bin/dolfinx-complex-mode
RUN chmod +x /usr/local/bin/dolfinx-*-mode

ONBUILD ARG MAKEFLAGS
ONBUILD ARG DOLFINX_CMAKE_BUILD_TYPE
ONBUILD ARG DOLFINX_CMAKE_CXX_FLAGS

ONBUILD WORKDIR /src

# This leaves the sources inside the container. This is a limitation of Docker.
# There is some trickery in the intermediate and dolfinx containers that can be
# used to remove this source if needed, see below.
ONBUILD ADD fiat/ /src/fiat/
ONBUILD ADD ufl/ /src/ufl/
ONBUILD ADD ffcx/ /src/ffcx/
ONBUILD ADD dolfinx/ /src/dolfinx/

# The dolfinx-onbuild container expects to have folders fiat/ ufl/ ffcx/ and
# dolfinx/ mounted/shared at /src.
ONBUILD RUN cd fiat && pip3 install --no-cache-dir . && \
    cd ../ufl && pip3 install --no-cache-dir . && \
    cd ../ffcx && pip3 install --no-cache-dir . && \
    cd ../ && pip3 install --no-cache-dir ipython

ONBUILD RUN cd dolfinx && \
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-real-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFINX_CMAKE_CXX_FLAGS} ../cpp && \
    ninja ${MAKEFLAGS} install && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-real-32 pip3 install --target /usr/local/dolfinx-real/lib/python3.8/dist-packages --no-dependencies . && \
    cd ../ && \
    rm -r build && \
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-complex-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFIN_CMAKE_CXX_FLAGS} ../cpp && \
    ninja ${MAKEFLAGS} install && \
    . /usr/local/dolfinx-complex/share/dolfinx/dolfinx.conf && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-complex-32 pip3 install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies . && \
    cd ../ && \
    rm -r build

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
LABEL description="DOLFIN-X in 32-bit real and complex modes"

# This layer manually copies the build artifacts from intermediate into dev-env
# to make the final image. This is a workaround for a well known limitation of
# Docker that you cannot cleanup after an ADD operation. This reduces the
# container size by around 80MB as the /src folder no longer exists in the final
# image.

COPY --from=intermediate /usr/local/dolfinx-real /usr/local/dolfinx-real
COPY --from=intermediate /usr/local/dolfinx-complex /usr/local/dolfinx-complex
COPY --from=intermediate /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=intermediate /usr/local/bin /usr/local/bin

# Real by default.
# Note that because we inherit from dev-env we do not inherit these ENV from
# dolfinx-onbuild.
ENV PKG_CONFIG_PATH=/usr/local/dolfinx-real/lib/pkgconfig:$PKG_CONFIG_PATH \
    PETSC_ARCH=linux-gnu-real-32 \
    PYTHONPATH=/usr/local/dolfinx-real/lib/python3.8/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-real/lib:$LD_LIBRARY_PATH

########################################

FROM dolfinx as lab
LABEL description="DOLFIN-X Jupyter Lab"

WORKDIR /root

RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab
EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
