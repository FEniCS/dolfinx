# Dockerfile describing development builds of FEniCS-X
#
# Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>

ARG GMSH_VERSION=4.5.5
ARG PYBIND11_VERSION=2.4.3
ARG PETSC_VERSION=3.12.4
ARG SLEPC_VERSION=3.12.2
ARG PETSC4PY_VERSION=3.12.0
ARG SLEPC4PY_VERSION=3.12.0

# Should be updated upon a new KaHIP release
ARG KAHIP_VERSION=14be06c

ARG PETSC_SLEPC_OPTFLAGS="-02 -g"
ARG PETSC_SLEPC_DEBUGGING="yes"

FROM ubuntu:20.04 as base
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"
LABEL description="Base image for real and complex FEniCS test environments"

ARG GMSH_VERSION
ARG PYBIND11_VERSION

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
# - Fourth set of packages are optional, required for meshio.
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
    libhdf5-openmpi-dev \
    liblapack-dev \
    libopenmpi-dev \
    libopenblas-dev \
    libscalapack-openmpi-dev \
    openmpi-bin \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-h5py \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-setuptools && \
    apt-get -y install \
    doxygen \
    git \
    sudo \
    valgrind \
    wget && \
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxinerama1 && \
    apt-get -y install \
    python3-lxml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages (via pip)
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
# - Third set of packages are optional but required for
#   pygmsh/meshio/DOLFINX mesh pipeline.
RUN python3 -m pip install --no-cache-dir mpi4py numba && \
    python3 -m pip install --no-cache-dir cffi cppimport flake8 pytest pytest-xdist sphinx sphinx_rtd_theme && \
    python3 -m pip install --no-cache-dir meshio pygmsh

# Install pybind11
RUN wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make install && \
    rm -rf /tmp/*

# # Install KaHIP
# RUN cd /usr/local && \
#     git clone https://github.com/schulzchristian/KaHIP.git && \
#     cd KaHIP/ && \
#     git checkout $KAHIP_VERSION && \
#     ./compile_withcmake.sh

# ENV KAHIP_ROOT=/usr/local/KaHIP

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

# Add gmsh python API
ENV PYTHONPATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/lib:$PYTHONPATH

########################################

FROM base as test
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"
LABEL description="FEniCS development environment"

ARG PETSC_VERSION
ARG SLEPC_VERSION
ARG PETSC_SLEPC_OPTFLAGS
ARG PETSC_SLEPC_DEBUGGING

WORKDIR /tmp

# Install PETSc with real and complex types, and 32 and 64 bit int
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    # wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    # mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    git clone --depth 1 --branch v${PETSC_VERSION} https://gitlab.com/petsc/petsc.git && \
    cd petsc && \
    # Real, 32-bit int
    python3 ./configure \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalapack=1 \
    --with-scalapack-lib=-lscalapack-openmpi \
    --with-scalar-type=real \
    --prefix=/usr/local/petsc && \
    make && \
    make install && \
    # Real, 64-bit int
    git clean -ffdx . && \
    python3 ./configure \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu_dist \
    --with-scalar-type=real \
    --with-64-bit-indices=1 \
    --prefix=/usr/local/petsc-64 && \
    make && \
    make install && \
    # Complex, 32-bit int
    git clean -ffdx . && \
    python3 ./configure \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=complex \
    --with-scalapack=1 \
    --with-scalapack-lib=-lscalapack-openmpi \
    --prefix=/usr/local/petsc-complex && \
    make && \
    make install && \
    # Complex, 64-bit int
    git clean -ffdx . && \
    python3 ./configure \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu_dist \
    --with-scalar-type=complex \
    --with-64-bit-indices=1 \
    --prefix=/usr/local/petsc-complex-64 && \
    make && \
    make install && \
    # Cleanup
    apt-get -y purge bison flex  && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install SLEPc with real and complex types, and 32 and 64 bit int
RUN git clone --depth 1 --branch v${SLEPC_VERSION} https://gitlab.com/slepc/slepc.git && \
    cd slepc \
    #
    export PETSC_DIR=/usr/local/petsc && \
    git clean -ffdx && \
    python3 ./configure --prefix=/usr/local/slepc && \
    make && \
    make install && \
    #
    export PETSC_DIR=/usr/local/petsc-64 && \
    git clean -ffdx && \
    python3 ./configure --prefix=/usr/local/slepc-64 && \
    make && \
    make install && \
    #
    export PETSC_DIR=/usr/local/petsc-complex && \
    git clean -ffdx && \
    python3 ./configure --prefix=/usr/local/slepc-complex && \
    make && \
    make install && \
    #
    export PETSC_DIR=/usr/local/petsc-complex-64 && \
    git clean -ffdx & \
    python3 ./configure --prefix=/usr/local/slepc-complex-64 && \
    make && \
    make install && \
    #
    rm -rf /tmp/*

########################################

FROM test as py-test-real
LABEL description="FEniCS development environment with PETSc complex mode"
ARG PETSC4PY_VERSION
ARG SLEPC4PY_VERSION
ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc
RUN python3 -m pip install --no-cache-dir petsc4py==${PETSC4PY_VERSION} && \
    python3 -m pip install --no-cache-dir slepc4py==${SLEPC4PY_VERSION}

########################################

FROM test as py-test-real-64
LABEL description="FEniCS development environment with PETSc complex mode"
ARG PETSC4PY_VERSION
ARG SLEPC4PY_VERSION
ENV PETSC_DIR=/usr/local/petsc-64 SLEPC_DIR=/usr/local/slepc-64
RUN python3 -m pip install --no-cache-dir petsc4py==${PETSC4PY_VERSION} && \
    python3 -m pip install --no-cache-dir slepc4py==${SLEPC4PY_VERSION}

########################################

FROM test as py-test-complex
LABEL description="FEniCS development environment with PETSc complex mode"
ARG PETSC4PY_VERSION
ARG SLEPC4PY_VERSION
ENV PETSC_DIR=/usr/local/petsc-complex SLEPC_DIR=/usr/local/slepc-complex
RUN python3 -m pip install --no-cache-dir petsc4py==${PETSC4PY_VERSION} && \
    python3 -m pip install --no-cache-dir slepc4py==${SLEPC4PY_VERSION}

########################################

FROM test as py-test-complex-64
LABEL description="FEniCS development environment with PETSc complex mode"
ARG PETSC4PY_VERSION
ARG SLEPC4PY_VERSION
ENV PETSC_DIR=/usr/local/petsc-complex-64 SLEPC_DIR=/usr/local/slepc-complex-64
RUN python3 -m pip install --no-cache-dir petsc4py==${PETSC4PY_VERSION} && \
    python3 -m pip install --no-cache-dir slepc4py==${SLEPC4PY_VERSION}
