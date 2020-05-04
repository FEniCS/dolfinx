# Dockerfile for DOLFINX-X CI testing
#
# Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>

ARG GMSH_VERSION=4.5.6
ARG PYBIND11_VERSION=2.5.0
ARG PETSC_VERSION=3.13.1
ARG SLEPC_VERSION=3.13.1
ARG PETSC4PY_VERSION=3.13.0
ARG SLEPC4PY_VERSION=3.13.0

# Should be updated upon a new KaHIP release
ARG KAHIP_VERSION=14be06c

ARG PETSC_SLEPC_OPTFLAGS="-02 -g"

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
    libboost-program-options-dev \
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
    #
    apt-get -y install \
    doxygen \
    git \
    valgrind \
    wget && \
    #
    apt-get -y install \
    libglu1 \
    libxcursor1 \
    libxinerama1 && \
    #
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

# Install KaHIP
RUN cd /usr/local && \
    git clone https://github.com/schulzchristian/KaHIP.git && \
    cd KaHIP/ && \
    git checkout $KAHIP_VERSION && \
    ./compile_withcmake.sh

ENV KAHIP_ROOT=/usr/local/KaHIP

# Download and install Gmsh SDK
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

ENV PETSC_DIR=/usr/local/petsc
ENV SLEPC_DIR=/usr/local/slepc

WORKDIR /tmp

# Install PETSc with real and complex types, and 32 and 64 bit integers
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
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-64-bit-indices=no \
    --with-debugging=yes \
    --with-fortran-bindings=no \
    --with-scalapack=yes \
    --with-scalapack-lib=-lscalapack-openmpi \
    --with-scalar-type=real \
    --with-shared-libraries && \
    make && \
    # Real, 64-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-real-64 \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --download-hypre \
    --download-mumps \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu_dist \
    --with-64-bit-indices=yes \
    --with-debugging=yes \
    --with-fortran-bindings=no \
    --with-scalapack=yes \
    --with-scalapack-lib=-lscalapack-openmpi \
    --with-scalar-type=real \
    --with-shared-libraries && \
    make && \
    # Complex, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-complex-32 \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-64-bit-indices=no \
    --with-debugging=yes \
    --with-fortran-bindings=no \
    --with-scalapack=yes \
    --with-scalapack-lib=-lscalapack-openmpi \
    --with-scalar-type=complex \
    --with-shared-libraries && \
    make && \
    # Complex, 64-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-complex-64 \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --download-hypre \
    --download-mumps \
    --download-ptscotch \
    --download-suitesparse \
    --download-superlu_dist \
    --with-64-bit-indices=yes \
    --with-debugging=yes \
    --with-fortran-bindings=no \
    --with-scalapack=yes \
    --with-scalapack-lib=-lscalapack-openmpi \
    --with-scalar-type=complex \
    --with-shared-libraries && \
    make && \
    # Cleanup
    apt-get -y purge bison flex  && \
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

# Install SLEPc with real and complex types, and 32 and 64 bit int
RUN wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p ${SLEPC_DIR} && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C ${SLEPC_DIR} --strip-components 1 && \
    cd ${SLEPC_DIR} && \
    #
    export PETSC_ARCH=linux-gnu-real-32 && \
    python3 ./configure && make && \
    #
    export PETSC_ARCH=linux-gnu-real-64 && \
    python3 ./configure && make && \
    #
    export PETSC_ARCH=linux-gnu-complex-32 && \
    python3 ./configure && make && \
    #
    export PETSC_ARCH=linux-gnu-complex-64 && \
    python3 ./configure && make && \
    # Cleanup
    rm -rf ${SLEPC_DIR}/CTAGS ${SLEPC_DIR}/TAGS ${SLEPC_DIR}/docs ${SLEPC_DIR}/src/ ${SLEPC_DIR}/**/obj/ ${SLEPC_DIR}/**/test/ && \
    rm -rf /tmp/*

ARG PETSC4PY_VERSION
ARG SLEPC4PY_VERSION
RUN export PETSC_ARCH=linux-gnu-real-32:linux-gnu-real-64:linux-gnu-complex-32:linux-gnu-complex-64 && \
    python3 -m pip install -v --no-cache petsc4py==${PETSC4PY_VERSION} && \
    python3 -m pip install -v --no-cache  slepc4py==${SLEPC4PY_VERSION}
