# Init autoconf
AC_INIT(dolfin, 0.5.2, dolfin@fenics.org)

# Init automake
AM_INIT_AUTOMAKE($PACKAGE, $VERSION, no-define)

# Tell automake not to generate Makefile.in unless asked to.
# This way the code can be installed on systems without automake installed.
AM_MAINTAINER_MODE

# Option --enable-debug
AC_ARG_ENABLE(debug,
              AC_HELP_STRING([--enable-debug],
                             [Turn on debugging and warnings (this is the default)]),
              enable_debug=yes)

# Option --enable-optimization
AC_ARG_ENABLE(optimization,
              AC_HELP_STRING([--enable-optimization],
                             [Turn on optimization]),
              enable_optimization=yes)

# Option --disable-curses
AC_ARG_ENABLE(curses,
              AC_HELP_STRING([--disable-curses],
                             [Compile without curses]),
              enable_curses=no)

# Option --with-petsc-dir
AC_ARG_WITH(petsc,
            AC_HELP_STRING([--with-petsc-dir=<path>],
                           [Specify path to PETSc]),
            [PETSC_DIR=$with_petsc_dir],)

# Option --with-petsc-bopt
AC_ARG_WITH(petsc,
            AC_HELP_STRING([--with-petsc-dir=<path>],
                           [Specify path to PETSc (O_c++ or g_c++)]),
            [BOPT=$with_petsc_bopt],)

# Option --with-petsc-version
AC_ARG_WITH(petsc-version,
            AC_HELP_STRING([--with-petsc-version=<version>],
                           [Specify PETSc version]),
            [PETSC_VERSION=$with_petsc_version],)

# Standard tests
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_LN_S
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T

# Make debug option default so developers don't forget it
echo "turning on debugging and warnings (use --enable-optimization to avoid this)."
enable_debug="yes"

# Handle option --enable-debug
if test x$enable_debug = xyes; then
    echo "enabling debugging compiler flags"
    CXXFLAGS='-DDEBUG=1 -g -O2 -Wall -Werror -ansi -std=c++98 '
    # We should also add -pedantic, but then we get an error when we try
    # to include mpio.h (long long not supported by ISO C++)
fi

# Handle option --enable-optimization
if test x$enable_optimization = xyes; then
    echo "enabling optimizing compiler flags"
    CXXFLAGS='-O3 -Wall -Werror -pedantic -ansi -std=c++98'
fi

# Handle option --disable-curses
if test x$disable_curses = x; then
    echo "enabling curses, use --disable-curses to disable"
    AC_SEARCH_LIBS(wbkgdset, ncurses,,
                   echo "*** Unable to find ncurses on your system. "
                   echo "*** Try ./configure --disable-curses or install ncurses." ; exit 1)
    CURSES_CFLAGS=''
    CURSES_LIBS='-lncurses'
else
    echo "disabling curses, use --enable-curses to enable"
    CURSES_CFLAGS='-DNO_CURSES=1'
    CURSES_LIBS=''
fi

# Handle option --with-petsc-dir=<path>
if test x$PETSC_DIR = x; then
    echo "assuming path to PETSc is /usr/local/lib/petsc"
    PETSC_DIR="/usr/local/lib/petsc/"
else
    echo "using PETSC_DIR=$PETSC_DIR"
fi

# Handle option --with-petsc-bopt=<bopt>
if test x$BOPT = x; then
    echo "assuming PETSc BOPT is g_c++"
    PETSC_DIR="g_c++"
else
    echo "using BOPT=$BOPT"
fi

# Handle option --with-petsc-version=<version>. If not supplied,
# we assume the version is 2.2.1. We can only handle 2.2.0 or 2.2.1.
if test x$PETSC_VERSION = x; then
    echo "assuming PETSc version is 2.2.1"
    PETSC_VERSION="2.2.1"
elif test $PETSC_VERSION = "2.2.0"; then
    echo "using PETSc version $PETSC_VERSION"
elif test $PETSC_VERSION = "2.2.1"; then
    echo "using PETSc version $PETSC_VERSION"
else
    echo "*** Can only handle PETSc version 2.2.0 or 2.2.1."
    exit 1
fi

# Check for PETSc
if test $PETSC_VERSION = "2.2.0"; then
    AC_CHECK_PROG(petsc_found, petscarch, yes, no)
    if test $petsc_found = yes; then
        PETSC_ARCH=`petscarch`
        #BOPT='O_c++'
        BOPT='g_c++'
        PETSC_CFLAGS="-I$PETSC_DIR/include -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=$PETSC_ARCH -DPETSC_USE_BOPT_g -DPETSC_USE_EXTERN_CXX -DPETSC_2_2_0"
        AC_SUBST(PETSC_ARCH)
        AC_SUBST(PETSC_DIR)
        AC_SUBST(BOPT)
        AC_SUBST(PETSC_CFLAGS)
        AC_SUBST(PETSC_VERSION)
    else
        echo "*** Unable to find PETSc on your system."
        echo "*** Use --with-petsc=<path> to specify the correct path for PETSc,"
        echo "*** or set the environment variable PETSC_DIR to the correct path."
        echo "*** Debian users need to install the package petsc-dev."
        exit 1
    fi
else
    AC_CHECK_PROG(petsc_found, configarch, yes, no, $PATH:$PETSC_DIR/bin)
    if test $petsc_found = yes; then
        # Extract PETSc variables from PETSC_DIR/bmake/common/variables
        PETSC_CFLAGS=`make -s -C $PETSC_DIR getincludedirs`
        PETSC_LIBS=`make -s -C $PETSC_DIR getlinklibs`
        # Extra flag to choose correct version
        PETSC_CFLAGS="$PETSC_CFLAGS -DPETSC_2_2_1"
    else
        echo "*** Unable to find PETSc on your system."
        echo "*** Use --with-petsc=<path> to specify the correct path for PETSc,"
        echo "*** or set the environment variable PETSC_DIR to the correct path."
        echo "*** PETSc is available at http://www-unix.mcs.anl.gov/petsc/petsc-2/"
        exit 1
    fi
fi

# Check for MPI (needed by PETSc)
AC_CHECK_PROG(mpi_found, mpirun, yes, no)
if test $mpi_found = yes; then
    # FIXME: Do correct test
    MPI_CFLAGS='-I/usr/include/mpi'
    AC_SUBST(MPI_CFLAGS)
else
    echo "*** Unable to find MPI on your system."
    echo "*** Perhaps you need to install the package mpich?"
    exit 1
fi

# Check for libxml2
AC_CHECK_PROG(libxml2_found, xml2-config, yes, no)
if test $libxml2_found = yes; then	  
    XML2_CFLAGS=`xml2-config --cflags`
    XML2_LIBS=`xml2-config --libs`
else
    echo "*** Unable to find libxml2 development files on your system."
    echo "*** Perhaps you need to install the package libxml2-dev?"
    exit 1
fi

# List of DOLFIN module libraries
# FIXME: Rename when old modules are removed
DOLFIN_MODULES_NEW="poisson"

# List of DOLFIN kernel libraries
DOLFIN_KERNEL="common element fem form function io la log main map math mesh ode quadrature settings common"

# Find modules
cd src/modules
./scanmodules.sh
cd ../..
DOLFIN_MODULES=""
for l in `cat src/modules/modules.list | grep -v "#"`; do
    DOLFIN_MODULES="$DOLFIN_MODULES$l "
done

# Generate include path for kernel
KERNEL_CFLAGS=""
DOLFIN_MAIN="main"
DIRS="$DOLFIN_MAIN $DOLFIN_KERNEL"
KERNEL_CFLAGS="-I\$(top_builddir)/src/modules "
for f in $DIRS; do
    KERNEL_CFLAGS="$KERNEL_CFLAGS-I\$(top_builddir)/src/kernel/$f "
done
KERNEL_CFLAGS="$KERNEL_CFLAGS $CURSES_CFLAGS $PETSC_CFLAGS $MPI_CFLAGS $XML2_CFLAGS"
AC_SUBST(KERNEL_CFLAGS)

# Generate include path for main (including also module code)
MAIN_CFLAGS=""
DIRS="$DOLFIN_MODULES $DOLFIN_MODULES_NEW"
for f in $DIRS; do
    MAIN_CFLAGS="$MAIN_CFLAGS-I\$(top_builddir)/src/modules/$f "
done
MAIN_CFLAGS="$KERNEL_CFLAGS $MAIN_CFLAGS"
AC_SUBST(MAIN_CFLAGS)

# Generate library list for linking in the correct order.
# FIXME: Not very pretty to list the libraries multiple times
DOLFIN_LIBS=""
DIRS="$DOLFIN_MAIN $DOLFIN_MODULES $DOLFIN_MODULES_NEW $DOLFIN_KERNEL $DOLFIN_KERNEL $DOLFIN_KERNEL $DOLFIN_KERNEL"
for f in $DIRS; do
    DOLFIN_LIBS="$DOLFIN_LIBS-ldolfin-$f "
done
DOLFIN_LIBS="$DOLFIN_LIBS $CURSES_LIBS $PETSC_LIBS $XML2_LIBS"
AC_SUBST(DOLFIN_LIBS)

# Export variables to makefiles
AC_SUBST(CXXFLAGS)
AC_SUBST(PACKAGE_VERSION)
AC_SUBST(PACKAGE_NAME)

# Create Makefiles Seems like automake cannot handle a variable
# containing the list of files, so this cannot be done
# automatically. We have to list all the files.
AC_OUTPUT( Makefile \
           src/Makefile \
           src/pre/Makefile \
           src/kernel/Makefile \
           src/kernel/common/Makefile \
           src/kernel/common/dolfin/Makefile \
           src/kernel/element/Makefile \
           src/kernel/element/dolfin/Makefile \
           src/kernel/fem/Makefile \
           src/kernel/fem/dolfin/Makefile \
           src/kernel/function/Makefile \
           src/kernel/function/dolfin/Makefile \
           src/kernel/form/Makefile \
           src/kernel/form/dolfin/Makefile \
           src/kernel/io/Makefile \
           src/kernel/io/dolfin/Makefile \
           src/kernel/la/Makefile \
           src/kernel/la/dolfin/Makefile \
           src/kernel/main/Makefile \
           src/kernel/main/dolfin/Makefile \
           src/kernel/map/Makefile \
           src/kernel/map/dolfin/Makefile \
           src/kernel/math/Makefile \
           src/kernel/math/dolfin/Makefile \
           src/kernel/mesh/Makefile \
           src/kernel/mesh/dolfin/Makefile \
           src/kernel/ode/Makefile \
           src/kernel/ode/dolfin/Makefile \
           src/kernel/quadrature/Makefile \
           src/kernel/quadrature/dolfin/Makefile \
           src/kernel/log/Makefile \
           src/kernel/log/dolfin/Makefile \
           src/kernel/settings/Makefile \
           src/kernel/settings/dolfin/Makefile \
           src/modules/Makefile \
           src/modules/dolfin/Makefile \
           src/modules/template/Makefile \
           src/modules/poisson/Makefile \
           src/modules/poisson/dolfin/Makefile \
           src/modules/poisson-multigrid/Makefile \
           src/modules/convdiff/Makefile \
           src/modules/convdiff/dolfin/Makefile \
           src/modules/euler/Makefile \
           src/modules/heat/Makefile \
           src/modules/heat/dolfin/Makefile \
           src/modules/navierstokes/Makefile \
           src/modules/navierstokes/dolfin/Makefile \
           src/modules/odesolver/Makefile \
           src/modules/wave/Makefile \
           src/modules/wave-vector/Makefile \
           src/modules/elasticity/Makefile \
           src/modules/elasticity-stationary/Makefile \
           src/modules/elasticity-updated/Makefile \
           src/config/Makefile \
           src/post/Makefile \
           src/demo/Makefile \
           src/demo/la/Makefile \
           src/demo/solvers/Makefile \
           src/demo/solvers/navierstokes/Makefile \
           src/demo/solvers/ode/Makefile \
           src/demo/solvers/ode/stiff/Makefile \
           src/demo/solvers/ode/mechanical/Makefile \
           src/demo/solvers/ode/modeling/Makefile \
           src/demo/solvers/ode/homotopy/Makefile \
           src/utils/Makefile \
           src/utils/inp2dx/Makefile \
           src/greeting/Makefile )
