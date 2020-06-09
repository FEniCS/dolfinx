# Dockerfile describing development builds of FEniCS-X
#
# Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>
#
# All layers are built nightly on CircleCI and pushed to
# https://hub.docker.com/orgs/dolfinx
#
# To run a nightly build:
#
#    docker run -ti dolfinx/dolfinx
#
# To run a Jupyter lab session:
#
#    docker run -p 8888:8888 dolfinx/lab
#
# To run and share the current host directory with the container:
#
#    docker run -p 8888:8888 -v "$(pwd)":/root/shared dolfinx/lab
#

ARG TINI_VERSION=v0.18.0
ARG GMSH_VERSION=4.5.5
ARG PYBIND11_VERSION=2.4.3
ARG PETSC_VERSION=3.12.4
ARG SLEPC_VERSION=3.12.2
ARG PETSC4PY_VERSION=3.12.0
ARG SLEPC4PY_VERSION=3.12.0
# Should be updated upon a new KaHIP release
ARG KAHIP_VERSION=14be06c

ARG MAKEFLAGS
ARG PETSC_SLEPC_OPTFLAGS="-02 -g"
ARG PETSC_SLEPC_DEBUGGING="yes"

########################################

FROM ubuntu:20.04 as dev-env
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"
LABEL description="FEniCS development environment with PETSc real and complex modes"

ARG GMSH_VERSION
ARG PYBIND11_VERSION
ARG PETSC_VERSION
ARG PETSC4PY_VERSION
ARG SLEPC_VERSION
ARG SLEPC4PY_VERSION

ARG MAKEFLAGS
ARG PETSC_SLEPC_OPTFLAGS
ARG PETSC_SLEPC_DEBUGGING

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

# Install Python packages (via pip)
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
# - Third set of packages are optional but required for
#   pygmsh/meshio/DOLFINX mesh pipeline.
RUN pip3 install --no-cache-dir mpi4py numba && \
    pip3 install --no-cache-dir cffi cppimport flake8 pytest pytest-xdist sphinx sphinx_rtd_theme && \
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

WORKDIR /tmp

# Install PETSc and SLEPc with real types
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    cd petsc-src && \
    python3 ./configure \
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
    --download-superlu_dist \
    --with-scalar-type=real \
    --prefix=/usr/local/petsc-real && \
    make ${MAKEFLAGS} && \
    make install && \
    export PETSC_DIR=/usr/local/petsc-real && \
    cd /tmp && \
    wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p slepc-src && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C slepc-src --strip-components 1 && \
    cd slepc-src && \
    python3 ./configure --prefix=/usr/local/slepc-real && \
    make ${MAKEFLAGS} && \
    make install && \
    apt-get -y purge bison flex python && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install petsc4py and slepc4py with real types
RUN PETSC_DIR=/usr/local/petsc-real SLEPC_DIR=/usr/local/slepc-real pip3 install --no-cache-dir petsc4py==${PETSC4PY_VERSION} slepc4py==${SLEPC4PY_VERSION}

# Install PETSc and SLEPc with complex scalar types
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    cd petsc-src && \
    python3 ./configure \
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
    --download-superlu_dist \
    --with-scalar-type=complex \
    --prefix=/usr/local/petsc-complex && \
    make ${MAKEFLAGS} && \
    make install && \
    export PETSC_DIR=/usr/local/petsc-complex && \
    cd /tmp && \
    wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p slepc-src && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C slepc-src --strip-components 1 && \
    cd slepc-src && \
    python3  ./configure --prefix=/usr/local/slepc-complex && \
    make ${MAKEFLAGS} && \
    make install && \
    apt-get -y purge bison flex python && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install petsc4py and slepc4py with complex types
RUN mkdir -p /usr/local/lib/python3.8/dist-packages-complex && \
    PETSC_DIR=/usr/local/petsc-complex SLEPC_DIR=/usr/local/slepc-real pip3 install --no-cache-dir --no-dependencies --ignore-installed --target /usr/local/lib/python3.8/dist-packages-complex petsc4py==${PETSC4PY_VERSION} slepc4py==${SLEPC4PY_VERSION}

# Default is to use real version of PETSc
# petsc4py and slepc4py build against real PETSc are found automatically in the default PYTHONPATH
# User must set PYTHONPATH to use complex version of petsc4py and slepc4py
ENV PETSC_DIR=/usr/local/petsc-real SLEPC_DIR=/usr/local/slepc-real

WORKDIR /root

########################################

FROM dev-env as dolfinx-onbuild
LABEL description="DOLFIN-X in real and complex modes (onbuild)"

ADD dolfinx-real-mode /usr/local/bin/dolfinx-real-mode
ADD dolfinx-complex-mode /usr/local/bin/dolfinx-complex-mode
RUN chmod +x /usr/local/bin/dolfinx-*-mode

ONBUILD ARG MAKEFLAGS

ONBUILD WORKDIR /tmp

# Install ipython (optional), FIAT, UFL and FFCX (development
# versions, master branch)
ONBUILD RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git

ONBUILD RUN git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja ${MAKEFLAGS} install && \
    cd ../python && \
    pip3 install . && \
    rm -rf /tmp/*

ONBUILD RUN git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    PYTHONPATH=/usr/local/lib/python3.8/dist-packages-complex PETSC_DIR=/usr/local/petsc-complex SLEPC_DIR=/usr/local/slepc-complex cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex ../cpp && \
    ninja ${MAKEFLAGS} install && \
    . /usr/local/dolfinx-complex/share/dolfinx/dolfinx.conf && \
    cd ../python && \
    PETSC_DIR=/usr/local/petsc-complex pip3 install --target /usr/local/lib/python3.8/dist-packages-complex --no-dependencies --ignore-installed . && \
    rm -rf /tmp/*

ONBUILD WORKDIR /root

########################################

FROM dolfinx-onbuild as dolfinx
LABEL description="DOLFIN-X in real and complex modes"

########################################

FROM dolfinx as lab
LABEL description="DOLFIN-X Jupyter Lab"
WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
