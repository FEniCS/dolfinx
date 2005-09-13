# Init autoconf
AC_INIT(dolfin, 0.5.8, dolfin@fenics.org)

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
              disable_curses=yes)

# Option --disable-mpi
AC_ARG_ENABLE(mpi,
              AC_HELP_STRING([--disalbe-mpi],
                             [Compile without MPI (assuming PETSc does not use MPI)]),
              disable_mpi=yes)

# Option --with-petsc-dir
AC_ARG_WITH(petsc-dir,
            AC_HELP_STRING([--with-petsc-dir=<path>],
                           [Specify path to PETSc]),
            [PETSC_DIR=$with_petsc_dir],)

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
    CXXFLAGS='-DDEBUG=1 -g -O2 -Wall -Werror -ansi -pedantic -std=c++98 '
    # We should also add -pedantic, but then we get an error when we try
    # to include mpio.h (long long not supported by ISO C++)
fi

# Handle option --enable-optimization
if test x$enable_optimization = xyes; then
    echo "enabling optimizing compiler flags"
    CXXFLAGS='-O3 -Wall -Werror -ansi -std=c++98'
fi

# Add '-U__STRICT_ANSI__'  flag if building under cygwin
if test `uname -o` = Cygwin; then
    echo "adding compiler flags for cygwin"
    CXXFLAGS=$CXXFLAGS' -U__STRICT_ANSI__ '
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
    echo "disabling curses"
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

# Check for PETSc
AC_CHECK_PROG(petsc_found, configarch, yes, no, $PATH:$PETSC_DIR/bin)
if test $petsc_found = yes; then
    # Extract PETSc variables from PETSC_DIR/bmake/common/variables
    PETSC_CFLAGS=`make -s -C $PETSC_DIR getincludedirs PETSC_DIR=$PETSC_DIR`
    PETSC_LIBS=`make -s -C $PETSC_DIR getlinklibs PETSC_DIR=$PETSC_DIR`
else
    echo "*** Unable to find PETSc 2.3.0 on your system."
    echo "*** Use --with-petsc-dir=<path> to specify the correct path for PETSc,"
    echo "*** or set the environment variable PETSC_DIR to the correct path."
    echo "*** PETSc is available at http://www-unix.mcs.anl.gov/petsc/petsc-2/"
    exit 1
fi

# Check for MPI (needed by PETSc)
if test x$disable_mpi = x; then
    AC_CHECK_PROG(mpi_found, mpirun, yes, no)
    if test $mpi_found = yes; then
        DIR=`which mpirun | sed 's/bin\/mpirun/include/'`
        if [ test -d $DIR/mpi ]; then
            MPI_CFLAGS="-I$DIR/mpi"
        else
	    MPI_CFLAGS="-I$DIR"
        fi
    else
        echo "*** Unable to find MPI on your system."
        echo "*** Perhaps you need to install the package mpich?"
        exit 1
    fi
else
    echo "disabling MPI (might not work if PETSc uses MPI)"
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
DOLFIN_MODULES="poisson convdiff navierstokes elasticity elasticity-updated"

# List of DOLFIN kernel libraries
DOLFIN_KERNEL="common fem form function io la log main math mesh ode quadrature settings common"

# Generate include path for kernel
KERNEL_CFLAGS=""
for f in $DOLFIN_KERNEL; do
    KERNEL_CFLAGS="$KERNEL_CFLAGS -I\$(top_builddir)/src/kernel/$f"
done
KERNEL_CFLAGS="$KERNEL_CFLAGS $CURSES_CFLAGS $PETSC_CFLAGS $MPI_CFLAGS $XML2_CFLAGS"
AC_SUBST(KERNEL_CFLAGS)

# Generate include path for modules
MODULE_CFLAGS="$KERNEL_CFLAGS -I\$(top_builddir)/src/modules/"
AC_SUBST(MODULE_CFLAGS)

# Generate include path for main (including also module code)
MAIN_CFLAGS=""
for f in $DOLFIN_MODULES; do
    MAIN_CFLAGS="$MAIN_CFLAGS -I\$(top_builddir)/src/modules/$f "
done
DOLFIN_CFLAGS="$KERNEL_CFLAGS $MAIN_CFLAGS"
AC_SUBST(DOLFIN_CFLAGS)

# Generate library list for linking in the correct order.
# FIXME: Not very pretty to list the libraries multiple times
DOLFIN_LIBS=""
DIRS="$DOLFIN_MODULES $DOLFIN_KERNEL $DOLFIN_KERNEL $DOLFIN_KERNEL $DOLFIN_KERNEL"
for f in $DIRS; do
    DOLFIN_LIBS="$DOLFIN_LIBS -ldolfin-$f"
done
DOLFIN_LIBS="$DOLFIN_LIBS $CURSES_LIBS $PETSC_LIBS $XML2_LIBS"
AC_SUBST(DOLFIN_LIBS)

# Export variables to makefiles
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
           src/modules/poisson/Makefile \
           src/modules/poisson/dolfin/Makefile \
           src/modules/convdiff/Makefile \
           src/modules/convdiff/dolfin/Makefile \
           src/modules/elasticity/Makefile \
           src/modules/elasticity/dolfin/Makefile \
           src/modules/elasticity-updated/Makefile \
           src/modules/elasticity-updated/dolfin/Makefile \
           src/modules/navierstokes/Makefile \
           src/modules/navierstokes/dolfin/Makefile \
           src/config/Makefile \
           src/post/Makefile \
           src/demo/Makefile \
           src/demo/bench/Makefile \
           src/demo/fem/Makefile \
           src/demo/solvers/Makefile \
           src/demo/solvers/navierstokes/Makefile \
           src/demo/solvers/ode/Makefile \
           src/demo/solvers/ode/modeling/Makefile \
           src/demo/solvers/ode/homotopy/Makefile \
           src/utils/Makefile \
           src/utils/inp2dx/Makefile \
           src/greeting/Makefile )
