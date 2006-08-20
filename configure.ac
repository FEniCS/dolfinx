# Init autoconf
AC_INIT(dolfin, 0.6.1, dolfin@fenics.org)

# Init automake
AM_INIT_AUTOMAKE($PACKAGE, $VERSION, no-define)

# Tell automake not to generate Makefile.in unless asked to.
# This way the code can be installed on systems without automake installed.
AM_MAINTAINER_MODE

# Option --enable/disable-debug
AC_ARG_ENABLE(debug,
              AC_HELP_STRING([--enable-debug],
                             [Turn on debugging and warnings (this is the default)]),
              enable_debug=$enableval, enable_debug=no)

# Option --enable/disable-optimization
AC_ARG_ENABLE(optimization,
              AC_HELP_STRING([--enable-optimization],
                             [Turn on optimization]),
              enable_optimization=$enableval, enable_optimization=no)

# Option --enable/disable-mpi
AC_ARG_ENABLE(mpi,
              AC_HELP_STRING([--enable-mpi],
                             [Compile with MPI]),
              enable_mpi=$enableval, enable_mpi=no)

# Option --enable/disable-pydolfin
AC_ARG_ENABLE(pydolfin,
              AC_HELP_STRING([--enable-pydolfin],
                             [Turn on compilation of PyDOLFIN]),
              enable_pydolfin=$enableval, enable_pydolfin=no)

# Option --enable/disable-curses
AC_ARG_ENABLE(curses,
              AC_HELP_STRING([--disable-curses],
                             [Compile without curses]),
              enable_curses=$enableval, enable_curses=yes)

# Option --enable/disable-petsc
AC_ARG_ENABLE(petsc,
              AC_HELP_STRING([--disable-petsc],
                             [Compile without PETSc (some demos may not compile)]),
              disable_petsc=$enableval, enable_petsc=yes)

# Option --with-petsc-dir
AC_ARG_WITH(petsc-dir,
            AC_HELP_STRING([--with-petsc-dir=<path>],
                           [Specify path to PETSc]),
            [PETSC_DIR=$with_petsc_dir],)

# Option --with-boost-include
AC_ARG_WITH(boost-include,
            AC_HELP_STRING([--with-boost-include=<path>],
                           [Specify path to Boost include files]),
            [BOOST_DIR=$with_boost_include],)

# Option --with-umfpack-include
AC_ARG_WITH(umfpack-include,
            AC_HELP_STRING([--with-umfpack-include=<path>],
                           [Specify path to umfpack include files]),
            [UMFPACK_INCLUDE_DIR=$with_umfpack_include],)

# Option --with-umfpack-lib
AC_ARG_WITH(umfpack-lib,
            AC_HELP_STRING([--with-umfpack-lib=<path>],
                           [Specify path to umfpack library]),
            [UMFPACK_LIB_DIR=$with_umfpack_lib],)

# Option --with-amd-lib
AC_ARG_WITH(amd-lib,
            AC_HELP_STRING([--with-amd-lib=<path>],
                           [Specify path to amd library (part of umfpack)]),
            [AMD_LIB_DIR=$with_amd_lib],)

# Standard tests
AC_PROG_CXX
AC_PROG_CC
AX_COMPILER_VENDOR
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_LN_S
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T

# Test that the libtool executable is installed
AC_CHECK_PROG(libtool_found, libtool, yes, no)
if test $libtool_found = no; then
    echo "*** Unable to find libtool."
    echo "*** Perhaps you need to install the package libtool?"
    exit 1	
fi    

# Configure libtool
AC_PROG_LIBTOOL

# Check for libxml2
AM_PATH_XML2( ,libxml2_found=yes, libxml2_found=no)
if test $libxml2_found = no; then
    echo "*** Unable to find libxml2 development files on your system."
    echo "*** Perhaps you need to install the packages libxml2 and libxml2-dev?"
    exit 1
fi

# Use C++ compiler for performing configure tests
AC_LANG(C++)

# Check for Boost FIXME: Check first user provided location, then default
AC_CHECK_HEADER([boost/version.hpp], have_boost=yes, have_boost=no)
if test $have_boost = no; then
    AC_CHECK_HEADER([$BOOST_DIR/boost/version.hpp], have_boost=yes)
    if test $have_boost = yes; then
        BOOST_CFLAGS="-I$BOOST_DIR"
    else
        echo "*** Unable to find Boost."
        echo "*** Perhaps you need set the include path --with-boost-include=<path> or to install the package libboost-dev?"
        exit 1
    fi
fi


# Check for umfpack
have_umfpack_header=yes
have_amd_lib=no
have_umfpack_and_amd_lib=no
# Check for amd library
if test x$AMD_LIB_DIR != x; then
    UMFPACK_LIBS="-L$AMD_LIB_DIR -lamd"
    have_amd_lib=yes
else
   AC_CHECK_LIB(amd, amd_info, UMFPACK_LIBS="-lamd" have_amd_lib=yes)
    if test $have_amd_lib != yes; then
        echo "Cannot find amd library (part of umfpack). Umfpack will be disabled"
    fi
fi
# Check for umfpack library if amd library has been found
if test $have_amd_lib = yes; then
    if test x$UMFPACK_LIB_DIR != x; then
        UMFPACK_LIBS="$UMFPACK_LIBS -L$UMFPACK_LIB_DIR -lumfpack"
        have_umfpack_and_amd_lib=yes
    else
        AC_CHECK_LIB(umfpack,umf_i_malloc, UMFPACK_LIBS="$UMFPACK_LIBS -lumfpack" have_umfpack_and_amd_lib=yes,,$UMFPACK_LIBS)
        if test $have_umfpack_and_amd_lib != yes; then
            echo "Cannot find umfpack library. Umfpack will be disabled"
        fi
    fi
fi
# Check for umfpack header files if umfpack and amd libraries have been found
if test $have_umfpack_and_amd_lib = yes; then
    if test x$UMFPACK_INCLUDE_DIR != x; then
        UMFPACK_CFLAGS="-I$UMFPACK_INCLUDE_DIR"
        AC_DEFINE(HAVE_UMFPACK_H,1)
        have_umfpack_header=yes
    else
        # Check various default locations  
        AC_CHECK_HEADER(umfpack.h, have_umfpack_header=yes, have_umfpack_header=no)
        if test $have_umfpack_header = yes; then
            AC_DEFINE(HAVE_UMFPACK_H,1)
        else
            AC_CHECK_HEADER(umfpack/umfpack.h, have_umfpack_header=yes, have_umfpack_header=no)
            if test $have_umfpack_header = yes; then
                AC_DEFINE(HAVE_UMFPACK_UMFPACK_H,1)
            else
                AC_CHECK_HEADER(ufsparse/umfpack.h, have_umfpack_header=yes, have_umfpack_header=no)
                if test $have_umfpack_header = yes; then
                    AC_DEFINE(HAVE_UFSPARSE_UMFPACK_H,1)    
                else
                    echo "Cannot find umfpack header files. Umfpack will be disabled"
                fi
            fi
       fi
    fi
fi

# Test for PyDOLFIN
if test x$enable_pydolfin = xyes; then
    AM_PATH_PYTHON(2.3)
    AC_PYTHON_DEVEL
    AC_PROG_SWIG(1.3.25)
    SWIG_ENABLE_CXX
    SWIG_PYTHON
fi

# Make debug option default so developers don't forget it
#echo "turning on debugging and warnings (use --enable-optimization to avoid this)."
#enable_debug="yes"

# Disable Boost debug mode for configure without debug/optimize options
CXXFLAGS="$CXXFLAGS -DNDEBUG"

# Handle option --enable-debug
if test x$enable_debug = xyes; then
    echo "enabling debugging compiler flags"
    CXXFLAGS="-DDEBUG=1 -g -O2 -Wall -Werror -ansi"
fi 

# Handle option --enable-optimization
if test x$enable_optimization = xyes; then
    echo "enabling optimizing compiler flags"
    CXXFLAGS="-O3  -Wall -ansi -DNDEBUG"
fi

# Add compiler specific flags
case "${ax_cv_c_compiler_vendor}" in
   gnu) 
        CXXFLAGS="$CXXFLAGS -pedantic -Wno-long-long -std=c++98"
        ;;
   intel)
        echo "WARNING: Intel C++ compiler version <= 9.0 is incompatible with 
        echo "header files gcc version <= 4.0". You can specify the version of 
        echo "gcc header files to be used by setting the compiler option 
        echo "-gcc-name=<name> through the environment variables"
        echo "e.g. setenv INTEL_CXXFLAGS -gcc-name=gcc-3.4"
        CXXFLAGS="$CXXFLAGS -wd383 -wd810 -wd869 -wd981 -wd1418 -wd1419 -wd1572 -wd1599 $INTEL_CXXFLAGS"
        ;;
esac

# Add '-Wno-uninitialized' due to a Cygwin gcc 3.4 bug.
if test `uname -o` = Cygwin; then
    echo "adding compiler flags for cygwin"
    CXXFLAGS="$CXXFLAGS -Wno-uninitialized"
fi

# Add '-fPIC' for AMD 64 bit system
if test `uname -m` = x86_64; then
    echo "adding -fPIC to compiler flags"
    CXXFLAGS="$CXXFLAGS -fPIC"
fi

# Handle option --disable-curses
if test x$disable_curses = x; then
    echo "enabling curses, use --disable-curses to disable"
    AC_SEARCH_LIBS(wbkgdset, ncurses,,
                   echo "*** Unable to find ncurses on your system. "
                   echo "*** Try ./configure --disable-curses or install ncurses." ; exit 1)
    CURSES_CFLAGS=""
    CURSES_LIBS="-lncurses"
else
    echo "disabling curses"
    CURSES_CFLAGS="-DNO_CURSES=1"
    CURSES_LIBS=""
fi
echo "curses libs $CURSES_LIBS"


# Handle option --enable-pydolfin
AM_CONDITIONAL(COMPILE_PYDOLFIN, test x$enable_pydolfin = xyes)
if test x$enable_pydolfin = xyes; then
    echo "enabling compilation of PyDOLFIN"
else
    echo "disabling compilation of PyDOLFIN"
fi

# Handle option --enable-mpi
if test x$enable_mpi = xyes; then
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
        echo "*** Perhaps you need to install the packages libmpich1.0-dev and mpich-bin?"
        exit 1
    fi
else
    echo "disabling MPI (might not work if PETSc uses MPI)"
fi

# Handle option --disable-petsc
if test x$disable_petsc = x; then
    
    # Handle option --with-petsc-dir=<path>
    if test x$PETSC_DIR = x; then
        PETSC_DIR="/usr/local/lib/petsc/"
        echo "assuming path to PETSc is $PETSC_DIR"
    else
        echo "using PETSC_DIR=$PETSC_DIR"
    fi

    # Check for PETSc
    MKFILE=$PETSC_DIR/bmake/common/base
    if test -f $MKFILE; then
        # Extract PETSc variables
        PETSC_CFLAGS=`make -f petsc.conf get_petsc_include`
        PETSC_LIBS=`make -f petsc.conf get_petsc_libs`
    else
        echo "*** Unable to find PETSc on your system."
        echo "*** Use --with-petsc-dir=<path> to specify the correct path for PETSc,"
        echo "*** or set the environment variable PETSC_DIR to the correct path."
        echo "*** PETSc is available at http://www-unix.mcs.anl.gov/petsc/petsc-2/"
        exit 1
    fi

    # Found PETSc, so set HAVE_PETSC_H
    PETSC_CFLAGS="-DHAVE_PETSC_H=1 $PETSC_CFLAGS"

    # Check for eigenvalue solver extension SLEPc. Variable SLEPC_DIR must be set.
    AC_CHECK_HEADER([$SLEPC_DIR/include/slepcversion.h], have_slepc=yes, have_slepc=no)

    if test $have_slepc = yes; then
        PETSC_ARCH=`cat $PETSC_DIR/bmake/petscconf | sed -e "s/PETSC_ARCH=//" -e "2d"`
        SLEPC_CFLAGS="-DHAVE_SLEPC_H=1 -I${SLEPC_DIR} -I${SLEPC_DIR}/include"
        SLEPC_LIBS="-L${SLEPC_DIR}/lib/${PETSC_ARCH} -lslepc"
        SLEPC_LIBS="-Wl,-rpath,${SLEPC_DIR}/lib/linux-gnu-cxx-real-debug $SLEPC_LIBS"
    else
        echo "SLEPc not found on your system."
        echo "  SLEPc is a PETSc extension that provides eigenvalue solvers."
        echo "  It is available at http://www.grycap.upv.es/slepc."
    fi
else
    # Compile without PETSc
    echo "disabling PETSc (some demos may not compile)"
    PETSC_CFLAGS=""
fi

# List of DOLFIN module libraries
DOLFIN_MODULES="elasticity elasticity-updated convdiff navierstokes poisson stokes heat"

# List of DOLFIN kernel libraries
DOLFIN_KERNEL="common fem form function io la main math mesh nls ode pde quadrature parameter log"

# Generate include path for kernel
KERNEL_CFLAGS=""
for f in $DOLFIN_KERNEL; do
    KERNEL_CFLAGS="$KERNEL_CFLAGS -I\$(top_builddir)/src/kernel/$f"
done
KERNEL_CFLAGS="$KERNEL_CFLAGS $CURSES_CFLAGS $PETSC_CFLAGS $SLEPC_CFLAGS $BOOST_CFLAGS $MPI_CFLAGS $XML_CPPFLAGS $UMFPACK_CFLAGS "
AC_SUBST(KERNEL_CFLAGS)

# Generate include path for modules
MODULE_CFLAGS="$KERNEL_CFLAGS -I\$(top_builddir)/src/modules/"
AC_SUBST(MODULE_CFLAGS)

# Generate include path for main (including also module code)
MAIN_CFLAGS=""
for f in $DOLFIN_MODULES; do
    MAIN_CFLAGS="$MAIN_CFLAGS -I\$(top_builddir)/src/modules/$f "
done
DOLFIN_CFLAGS="$KERNEL_CFLAGS $MAIN_CFLAGS $MODULE_CFLAGS"
AC_SUBST(DOLFIN_CFLAGS)

# External libraries which DOLFIN depends on
DOLFIN_EXT_LIBS="$CURSES_LIBS $PETSC_LIBS $SLEPC_LIBS $XML_LIBS $UMFPACK_LIBS "
AC_SUBST(DOLFIN_EXT_LIBS)

# Library list which DOLFIN applications need to link against
DOLFIN_LIBS="-ldolfin"
DOLFIN_LIBS="$DOLFIN_LIBS $DOLFIN_EXT_LIBS"
AC_SUBST(DOLFIN_LIBS)

# Generate libtool library list (including also module code)
DOLFIN_LTLIBOBJS=""
for f in $DOLFIN_MODULES; do
    DOLFIN_LTLIBOBJS="$DOLFIN_LTLIBOBJS \$(top_builddir)/src/modules/$f/*.lo "
done
for f in $DOLFIN_KERNEL; do
    DOLFIN_LTLIBOBJS="$DOLFIN_LTLIBOBJS \$(top_builddir)/src/kernel/$f/*.lo "
done
AC_SUBST(DOLFIN_LTLIBOBJS)

# Export variables to makefiles
AC_SUBST(PACKAGE_VERSION)
AC_SUBST(PACKAGE_NAME)

# Create Makefiles, seems like automake cannot handle a variable
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
           src/kernel/nls/Makefile \
           src/kernel/nls/dolfin/Makefile \
           src/kernel/ode/Makefile \
           src/kernel/ode/dolfin/Makefile \
           src/kernel/pde/Makefile \
           src/kernel/pde/dolfin/Makefile \
           src/kernel/parameter/Makefile \
           src/kernel/parameter/dolfin/Makefile \
           src/kernel/quadrature/Makefile \
           src/kernel/quadrature/dolfin/Makefile \
           src/kernel/log/Makefile \
           src/kernel/log/dolfin/Makefile \
           src/modules/Makefile \
           src/modules/dolfin/Makefile \
           src/modules/convdiff/Makefile \
           src/modules/convdiff/dolfin/Makefile \
           src/modules/elasticity/Makefile \
           src/modules/elasticity/dolfin/Makefile \
           src/modules/elasticity-updated/Makefile \
           src/modules/elasticity-updated/dolfin/Makefile \
           src/modules/navierstokes/Makefile \
           src/modules/navierstokes/dolfin/Makefile \
           src/modules/poisson/Makefile \
           src/modules/poisson/dolfin/Makefile \
           src/modules/heat/Makefile \
           src/modules/heat/dolfin/Makefile \
           src/modules/stokes/Makefile \
           src/modules/stokes/dolfin/Makefile \
           src/lib/Makefile \
           src/pydolfin/Makefile \
           src/config/Makefile \
           src/post/Makefile \
           src/demo/Makefile \
           src/demo/fem/Makefile \
           src/demo/nls/Makefile \
           src/demo/ode/Makefile \
           src/demo/ode/homotopy/Makefile \
           src/demo/pde/Makefile \
           src/demo/pde/nonlinear/Makefile \
           src/demo/pde/stokes/Makefile \
           src/demo/solvers/Makefile \
           src/demo/solvers/navierstokes/Makefile \
           src/demo/scripting/Makefile \
           src/bench/Makefile \
           src/bench/fem/Makefile \
           src/bench/la/Makefile \
           src/bench/ode/Makefile \
           src/utils/Makefile \
           src/utils/inp2dx/Makefile \
           src/utils/convert/Makefile \
           src/greeting/Makefile )

# Display some configuration options
echo "---------------------------------------------------------"
echo "Configuration of DOLFIN finished. Now type"
echo ""
echo "   make"
echo "   make install"
echo ""
echo "to install DOLFIN on your system. After DOLFIN has been"
echo "installed, you can compile all the demo programs in the"
echo "subdirectory src/demo by running"
echo ""
echo "   make demo"
echo ""
echo "Note that you may need to be root in order to install."
echo "To specify an alternative installation directory, rerun"
echo "configure with option --prefix=<path>. You may also run"
echo "./configure.local to configure for a local installation"
echo "in the DOLFIN source tree."
echo "---------------------------------------------------------"

# Check that the installation directory is set up correctly
__BINDIR=`echo $prefix/bin | sed -e 's/\//-/g'`
__PATH=`echo $PATH | sed -e 's/\//-/g'`
if test "$__PATH" = "`echo $__PATH | sed s/$__BINDIR//`"; then
   echo 'Warning: installation directory is not in PATH.'
   echo ''
   echo 'To compile a program against DOLFIN (including the demos)'
   echo 'you need to add the bin subdirectory of the installation'
   echo 'diretory to your PATH which you can do with the command'
   echo ''
   echo "  export PATH=\"$prefix/bin:\$PATH\""
   echo ''
   echo 'for bash or, alternatively,'
   echo ''
   echo "  setenv PATH $prefix/bin:\${PATH}"
   echo ''
fi

# Hint for PyDOLFIN when installed locally
if test x$enable_pydolfin = xyes; then
   echo 'To use the DOLFIN Python interface (PyDOLFIN), you may need'
   echo 'set your PYTHONPATH which you can do with the command'
   echo ''
   echo "  export PYTHONPATH=\"$prefix/lib/python2.4/site-packages:\$PYTHONPATH\""
   echo ''
   echo 'for bash or, alternatively,'
   echo ''
   echo "  setenv PYTHONPATH $prefix/lib/python2.4/site-packages:\${PYTHONPATH}"
   echo ''
fi
