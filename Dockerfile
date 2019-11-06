# Dockerfile describing development environments and builds of FEniCS-X
#
# Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>
#
# All layers are built bi-weekly on CircleCI and pushed to
# https://quay.io/repository/fenicsproject/dolfinx
#
# To build development environment images:
#
#    docker build --target dev-env-complex -t quay.io/fenicsproject/dolfinx:dev-env-complex .
#    docker build --target dev-env-real -t quay.io/fenicsproject/dolfinx:dev-env-real .
#
# To run a notebook:
#
#    docker run -p 8888:8888 quay.io/fenicsproject/dolfinx:notebook
#
# To run and share the current host directory with the container:
#
#    docker run -p 8888:8888 -v "$(pwd)":/tmp quay.io/fenicsproject/dolfinx:notebook
#

ARG GMSH_VERSION=4.4.1
ARG PYBIND11_VERSION=2.4.3
ARG PETSC_VERSION=3.12
ARG SLEPC_VERSION=3.12.0
ARG PETSC4PY_VERSION=3.12.0
ARG SLEPC4PY_VERSION=3.12.0
ARG TINI_VERSION=v0.18.0

ARG MAKEFLAGS
ARG PETSC_SLEPC_OPTFLAGS="-02 -g"
ARG PETSC_SLEPC_DEBUGGING="yes"

FROM ubuntu:18.04 as base
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
        libhdf5-mpich-dev \
        liblapack-dev \
        libmpich-dev \
        libopenblas-dev \
        mpich \
        ninja-build \
        pkg-config \
        python3-dev \
        python3-matplotlib \
        python3-numpy \
        python3-pip \
        python3-scipy \
        python3-setuptools && \
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
    apt-get -y install \
        python3-lxml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz
ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

# Add gmsh python API
ENV PYTHONPATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/lib

# Install Python packages (via pip)
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
# - Third set of packages are optional but required for
#   pygmsh/meshio/DOLFIN mesh pipeline.
RUN pip3 install --no-cache-dir mpi4py numba && \
    pip3 install --no-cache-dir cffi flake8 pytest pytest-xdist sphinx sphinx_rtd_theme && \
    export HDF5_MPI="ON" && \
    pip3 install --no-cache-dir --no-binary=h5py h5py meshio pygmsh
# Install pybind11
RUN wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make install && \
    rm -rf /tmp/*

WORKDIR /root

########################################

FROM base as dev-env-real
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"
LABEL description="FEniCS development environment with PETSc real mode"

ARG PETSC_VERSION
ARG PETSC4PY_VERSION
ARG SLEPC_VERSION
ARG SLEPC4PY_VERSION

ARG MAKEFLAGS
ARG PETSC_SLEPC_OPTFLAGS
ARG PETSC_SLEPC_DEBUGGING

WORKDIR /tmp

# Install PETSc and SLEPc with real types.
RUN apt-get -qq update && \
    apt-get -y install bison flex python && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    cd petsc-src && \
    ./configure \
        --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --with-debugging=${PETSC_SLEPC_DEBUGGING} \
        --with-fortran-bindings=no \
        --download-blacs \
        --download-hypre \
        --download-metis \
        --download-mumps \
        --download-ptscotch \
        --download-scalapack \
        --download-spai \
        --download-suitesparse \
        --download-superlu \
        --with-scalar-type=real \
        --prefix=/usr/local/petsc && \
    make ${MAKEFLAGS} && \
    make install && \
    export PETSC_DIR=/usr/local/petsc && \
    cd /tmp && \
    wget -nc --quiet http://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p slepc-src && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C slepc-src --strip-components 1 && \
    cd slepc-src && \
    ./configure --prefix=/usr/local/slepc && \
    make ${MAKEFLAGS} && \
    make install && \
    apt-get -y purge bison flex python && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc

# Install petsc4py and slepc4py
RUN pip3 install --no-cache-dir petsc4py==${PETSC4PY_VERSION} && \
    pip3 install --no-cache-dir slepc4py==${SLEPC4PY_VERSION}

WORKDIR /root

########################################

FROM base as dev-env-complex
LABEL description="FEniCS development environment with PETSc complex mode"

ARG PETSC_VERSION
ARG PETSC4PY_VERSION
ARG SLEPC_VERSION
ARG SLEPC4PY_VERSION

ARG MAKEFLAGS
ARG PETSC_SLEPC_OPTFLAGS
ARG PETSC_SLEPC_DEBUGGING

WORKDIR /tmp

# Install PETSc and SLEPc with complex scalar types
RUN apt-get -qq update && \
    apt-get -y install bison flex python && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    cd petsc-src && \
    ./configure \
        --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --with-debugging=${PETSC_SLEPC_DEBUGGING} \
        --with-fortran-bindings=no \
        --download-blacs \
        --download-metis \
        --download-mumps \
        --download-ptscotch \
        --download-scalapack \
        --download-suitesparse \
        --download-superlu \
        --with-scalar-type=complex \
        --prefix=/usr/local/petsc && \
    make ${MAKEFLAGS} && \
    make install && \
    export PETSC_DIR=/usr/local/petsc && \
    cd /tmp && \
    wget -nc --quiet http://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p slepc-src && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C slepc-src --strip-components 1 && \
    cd slepc-src && \
    ./configure --prefix=/usr/local/slepc && \
    make ${MAKEFLAGS} && \
    make install && \
    apt-get -y purge bison flex python && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc

# Install complex petsc4py and slepc4py
RUN pip3 install --no-cache-dir petsc4py==${PETSC4PY_VERSION} && \
    pip3 install --no-cache-dir slepc4py==${SLEPC4PY_VERSION}

WORKDIR /root

########################################

FROM dev-env-real as real
LABEL description="DOLFIN-X in real mode"

ARG MAKEFLAGS

WORKDIR /tmp

# Install ipython (optional), FIAT, UFL and ffcX (development
# versions, master branch)
RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/fenics/ffcX

# Install dolfinx
RUN git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja ${MAKEFLAGS} install && \
    cd ../python && \
    pip3 install . && \
    rm -rf /tmp/*

WORKDIR /root

########################################

FROM dev-env-complex as complex
LABEL description="DOLFIN-X in complex mode"

ARG MAKEFLAGS

WORKDIR /tmp

# Install ipython (optional), FIAT, UFL and ffcX (development versions,
# master branch)
RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/fenics/ffcX

# Install dolfinx
RUN git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja ${MAKEFLAGS} install && \
    cd ../python && \
    pip3 install . && \
    rm -rf /tmp/*

WORKDIR /root

########################################

FROM real as notebook
LABEL description="DOLFIN-X Jupyter Notebook"
WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab

ENTRYPOINT ["/tini", "--", "jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

########################################

FROM complex as notebook-complex
LABEL description="DOLFIN-X (complex mode) Jupyter Notebook"

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab

WORKDIR /root
ENTRYPOINT ["/tini", "--", "jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

########################################

FROM notebook as lab
LABEL description="DOLFIN-X Jupyter Lab"

WORKDIR /root
ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

########################################

FROM notebook-complex as lab-complex
LABEL description="DOLFIN-X (complex mode) Jupyter Lab"

WORKDIR /root
ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
