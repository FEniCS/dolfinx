# Dockerfile describing development builds of FEniCS-X
#
# Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>
#
# All layers are built nightly on CircleCI and pushed to
# https://quay.io/repository/fenicsproject/dolfinx
#
# To run a nightly build:
#
#    docker run -ti quay.io/fenicsproject/dolfinx:real
#    docker run -ti quay.io/fenicsproject/dolfinx:complex
#
# To build a new FEniCS-X image from master (real or complex version):
#
#    docker build --target real-onbuild .
#    docker build --target complex-onbuild .
#
# To run a notebook:
#
#    docker run -p 8888:8888 quay.io/fenicsproject/dolfinx:notebook
#
# To run and share the current host directory with the container:
#
#    docker run -p 8888:8888 -v "$(pwd)":/tmp quay.io/fenicsproject/dolfinx:notebook
#

ARG TINI_VERSION=v0.18.0

FROM quay.io/fenicsproject/dolfinx:dev-env-real as real-onbuild
LABEL description="DOLFIN-X in real mode (ONBUILD)"

ONBUILD ARG MAKEFLAGS

ONBUILD WORKDIR /tmp

# Install ipython (optional), FIAT, UFL and ffcX (development
# versions, master branch)
ONBUILD RUN pip3 install --no-cache-dir ipython && \
            pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
            pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
            pip3 install --no-cache-dir git+https://github.com/fenics/ffcX

# Install dolfinx
ONBUILD RUN git clone https://github.com/fenics/dolfinx.git && \
            cd dolfinx && \
            mkdir build && \
            cd build && \
            cmake -G Ninja ../cpp && \
            ninja ${MAKEFLAGS} install && \
            cd ../python && \
            pip3 install . && \
            rm -rf /tmp/*

ONBUILD WORKDIR /root

########################################

FROM real-onbuild as real
LABEL description="DOLFIN-X in real mode"

########################################

FROM quay.io/fenicsproject/dolfinx:dev-env-complex as complex-onbuild
LABEL description="DOLFIN-X in complex mode (ONBUILD)"

ONBUILD ARG MAKEFLAGS

ONBUILD WORKDIR /tmp

# Install ipython (optional), FIAT, UFL and ffcX (development versions,
# master branch)
ONBUILD RUN pip3 install --no-cache-dir ipython && \
            pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
            pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
            pip3 install --no-cache-dir git+https://github.com/fenics/ffcX

# Install dolfinx
ONBUILD RUN git clone https://github.com/fenics/dolfinx.git && \
            cd dolfinx && \
            mkdir build && \
            cd build && \
            cmake -G Ninja ../cpp && \
            ninja ${MAKEFLAGS} install && \
            cd ../python && \
            pip3 install . && \
            rm -rf /tmp/*

ONBUILD WORKDIR /root

########################################

FROM complex-onbuild as complex
LABEL description="DOLFIN-X in real mode"

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
