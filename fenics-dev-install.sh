#!/usr/bin/env bash
#
# Developer script for configure + build + rebuild.
#
# Notes:
#
# - This script is what most developers use to build/rebuild this package.
# - This script works for both CMake and distutils based packages.
# - If this script is updated in one package, please propagate to the others!
#
# Environment variables:
#
# - $PROCS                    : controls number of processes to use for build
#                             : defaults to 6
# - $FENICS_PYTHON_EXECUTABLE : name of python executable
#                             : defaults to "python"
# - $FENICS_INSTALL_PREFIX    : path to FEniCS installation prefix
#                             : defaults to "${HOME}/opt/<branchname>"

# Exit on first error
set -e

# Get branch name
BRANCH=`(git symbolic-ref --short HEAD 2> /dev/null || git describe HEAD) | sed s:/:.:g`
echo "On branch '${BRANCH}'."

# Get installation prefix
: ${FENICS_INSTALL_PREFIX:="${HOME}/opt/fenics/${BRANCH}"}
echo "Installation prefix set to '${FENICS_INSTALL_PREFIX}'."

# Get Python executable and version
: ${FENICS_PYTHON_EXECUTABLE:=python}
FENICS_PYTHON_VERSION=$(${FENICS_PYTHON_EXECUTABLE} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python executable and version set to '${FENICS_PYTHON_EXECUTABLE} ${FENICS_PYTHON_VERSION}'."

# Get number of processes to use for build
: ${PROCS:=6}


# Build and install distutils based FEniCS package
if [ -e setup.py ]; then
    ${FENICS_PYTHON_EXECUTABLE} setup.py build
    ${FENICS_PYTHON_EXECUTABLE} setup.py install --prefix=${FENICS_INSTALL_PREFIX}
fi


# Build and install dolfin
if [ -e CMakeLists.txt ]; then
    # Set build directory
    if [ "${BRANCH}" = "master" ]; then
        BUILD_DIR=build.${BRANCH}
    elif [ "${BRANCH}" = "next" ]; then
        BUILD_DIR=build.${BRANCH}
    else
        BUILD_DIR="build.wip" # use for all other branches to save disk space
    fi

    # Configure
    CMAKE_EXTRA_ARGS=$@
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}
    time cmake -DCMAKE_INSTALL_PREFIX=${FENICS_INSTALL_PREFIX} \
               -DDOLFIN_ENABLE_TESTING=true \
               -DDOLFIN_ENABLE_BENCHMARKS=true \
               -DCMAKE_BUILD_TYPE=Developer \
               -DDOLFIN_DEPRECATION_ERROR=false \
               ${CMAKE_EXTRA_ARGS} \
               ..

    # Build and install
    time make -j ${PROCS} -k && make install -j ${PROCS}
fi


# Write config file
CONFIG_FILE="${FENICS_INSTALL_PREFIX}/fenics.conf"
rm -f ${CONFIG_FILE}
cat << EOF > ${CONFIG_FILE}
# FEniCS configuration file created by fenics-dev-install.sh on $(date)
export FENICS_INSTALL_PREFIX=${FENICS_INSTALL_PREFIX}
export FENICS_PYTHON_EXECUTABLE=${FENICS_PYTHON_EXECUTABLE}
export FENICS_PYTHON_VERSION=${FENICS_PYTHON_VERSION}

# Source FEniCS dependencies if found
FENICS_DEPS_CONF=\${HOME}/opt/fenics/fenics.deps
if [ -e \${FENICS_DEPS_CONF} ]; then
    source \${FENICS_DEPS_CONF}
fi

# Common Unix variables
export LD_LIBRARY_PATH=\${FENICS_INSTALL_PREFIX}/lib:\${LD_LIBRARY_PATH}
export PATH=\${FENICS_INSTALL_PREFIX}/bin:\${PATH}
export PKG_CONFIG_PATH=\${FENICS_INSTALL_PREFIX}/pkgconfig:\${PKG_CONFIG_PATH}
export PYTHONPATH=\${FENICS_INSTALL_PREFIX}/lib/python${FENICS_PYTHON_VERSION}/site-packages:\${PYTHONPATH}
export MANPATH=\${FENICS_INSTALL_PREFIX}/share/man:\${MANPATH}

# Set Instant cache modules separately for each install
export INSTANT_CACHE_DIR=\${FENICS_INSTALL_PREFIX}/cache/instant

# CMake search path
export CMAKE_PREFIX_PATH=\${FENICS_INSTALL_PREFIX}:\${CMAKE_PREFIX_PATH}
EOF
if [ $(uname) = "Darwin" ]; then
    cat << EOF >> $CONFIG_FILE

# Mac specific path
export DYLD_FALLBACK_LIBRARY_PATH=\${FENICS_INSTALL_PREFIX}/lib:\${DYLD_FALLBACK_LIBRARY_PATH}
EOF
fi

# Print information
echo
echo "- Installed branch '${BRANCH}' to ${FENICS_INSTALL_PREFIX}."
echo
echo "- Config file written to ${CONFIG_FILE}"
echo "  (source this file)."
echo
