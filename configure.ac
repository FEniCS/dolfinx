# Init autoconf
AC_INIT(DOLFIN, 0.7.1, http://www.fenics.org/bugzilla/)

# Init automake
AM_INIT_AUTOMAKE($PACKAGE_NAME, $PACKAGE_VERSION, no-define)

# Tell automake not to generate Makefile.in unless asked to.
# This way the code can be installed on systems without automake installed.
AM_MAINTAINER_MODE

# Option --enable/disable-debug
AC_ARG_ENABLE(debug,
              AC_HELP_STRING([--enable-debug],
                             [Turn on debugging and warnings]),
              enable_debug=$enableval, enable_debug=no)

# Option --enable/disable-debug-ublas
AC_ARG_ENABLE(debug-ublas,
              AC_HELP_STRING([--enable-debug-ublas],
                             [Turn on extra debugging for uBlas, only works with --enable-debug]),
              enable_debug_ublas=$enableval, enable_debug_ublas=no)

# Option --enable/disable-optimization
AC_ARG_ENABLE(optimization,
              AC_HELP_STRING([--enable-optimization],
                             [Turn on optimizations]),
              enable_optimization=$enableval, enable_optimization=no)

# Option --enable/disable-pydolfin
AC_ARG_ENABLE(pydolfin,
              AC_HELP_STRING([--disable-pydolfin],
                             [Disable compilation of PyDOLFIN, the DOLFIN Python interface]),
              enable_pydolfin=$enableval, enable_pydolfin=yes)

# Option --enable/disable-petsc
AC_ARG_ENABLE(petsc,
              AC_HELP_STRING([--enable-petsc],
                             [Compile with support for PETSc linear algebra]),
              disable_petsc=$enableval, enable_petsc=no)

# Option --with-petsc-dir
AC_ARG_WITH(petsc-dir,
            AC_HELP_STRING([--with-petsc-dir=<path>],
                           [Specify path to PETSc]),
            [PETSC_DIR=$with_petsc_dir],)

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

# Option --enable/disable-scotch
AC_ARG_ENABLE(scotch,
              AC_HELP_STRING([--enable-scotch],
                             [Compile with support for SCOTCH graph partitioning]),
              disable_scotch=$enableval, enable_scotch=no)

# Use C++ compiler for performing configure tests
AC_LANG(C++)

# Check for MPI and print message if not found
AC_CHECK_PROG(mpi_found, mpirun, yes, no)
if test $mpi_found = no; then
    echo "MPI not found (might not work if PETSc uses MPI)."
else
    # Found MPI, so set HAVE_MPI_H and IGNORE_CXX_SEEK (mpich2 bug)
    CXXFLAGS="-DHAVE_MPI_H=1 -DMPICH_IGNORE_CXX_SEEK $CXXFLAGS"
fi

# Standard tests
AC_PROG_CXX([mpic++ mpicxx c++ g++])
AC_PROG_CC([mpicc cc gcc])
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

# Check for Boost
AX_BOOST_BASE([1.33])

# Check for Glib
PKG_CHECK_MODULES(GLIB, glib-2.0, have_glib=yes, have_glib=no)

# Check for GTS
PKG_CHECK_MODULES(GTS, gts, have_gts=yes, have_gts=no)
if test $have_gts = yes && test $have_glib = yes; then
    GTS_CFLAGS="$GTS_CFLAGS -DHAVE_GTS_H=1"
else
    echo "Missing GTS"
fi

# Check for UFC
PKG_CHECK_MODULES(UFC, ufc-1)

# Prepare for AMD and UMFPACK tests
have_umfpack_header=no
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
#if test $have_amd_lib = yes; then
#    if test x$UMFPACK_LIB_DIR != x; then
#        UMFPACK_LIBS="$UMFPACK_LIBS -L$UMFPACK_LIB_DIR -lumfpack"
#        have_umfpack_and_amd_lib=yes
#    else
#        AC_CHECK_LIB(umfpack, umfpack_dl_solve, UMFPACK_LIBS="$UMFPACK_LIBS -lumfpack" have_umfpack_and_amd_lib=yes,,$UMFPACK_LIBS)
#        if test $have_umfpack_and_amd_lib != yes; then
#            echo "Cannot find umfpack library. Umfpack will be disabled"
#        fi
#    fi
#fi

# Check for umfpack header files if umfpack and amd libraries have been found
#if test $have_umfpack_and_amd_lib = yes; then
if test $have_amd_lib = yes; then
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
                    AC_CHECK_HEADER(suitesparse/umfpack.h, have_umfpack_header=yes, have_umfpack_header=no)
                    if test $have_umfpack_header = yes; then
                        AC_DEFINE(HAVE_SUITESPARSE_UMFPACK_H,1)    
                    else
                        echo "Cannot find umfpack header files. Umfpack will be disabled"
                    fi
                fi
            fi
       fi
    fi
fi

# This is a temporary fix to support both libumfpack4 and libufsparse.
# In libufsparse, -lblas is missing, so we skip the test for the library
# above, only check for the header, and add the flag here.
if test $have_umfpack_header = yes; then
   if test x$UMFPACK_LIB_DIR != x; then
      UMFPACK_LIBS="$UMFPACK_LIBS -L$UMFPACK_LIB_DIR -lumfpack -lblas"
   else
      UMFPACK_LIBS="$UMFPACK_LIBS -lumfpack -lblas"
   fi
fi

# Test for PyDOLFIN
if test x$enable_pydolfin = xyes; then
    AM_PATH_PYTHON(2.3)
    AC_PYTHON_DEVEL
    CPPFLAGS="$CPPFLAGS $PYTHON_CPPFLAGS"
    LDFLAGS="$LDFLAGS $PYTHON_LDFLAGS $PYTHON_EXTRA_LIBS"
    AC_CHECK_HEADER(Python.h, have_python_dev=yes, have_python_dev=no)
    if test $have_python_dev = no; then
       echo "*** Cannot find Python development files on your system."
       echo "*** Perhaps you need to install the package python$PYTHON_VERSION-dev?"
       exit 1
    fi

    AC_CHECK_PROG(swig_found, swig, yes, no)
    if test $swig_found = no; then
      echo "*** Unable to find SWIG."
      echo "*** Install SWIG or disable PyDOLFIN with --disable-pydolfin"
      exit 1	
    fi    

    AC_PROG_SWIG(1.3.25)

    SWIG_ENABLE_CXX
    SWIG_PYTHON

fi

# Check for NumPy
if test x$enable_pydolfin = xyes; then
    echo "checking for NumPy..."
    NUMPY_CFLAGS=`$PYTHON numpy.conf`
    if test $NUMPY_CFLAGS == none; then
       echo "*** Cannot find NumPy on your system."
       echo "*** Perhaps you need to install the package python-numpy?"
       exit 1
    fi
    CPPFLAGS="$CPPFLAGS -I$NUMPY_CFLAGS"
fi

# Make debug option default so developers don't forget it
#echo "turning on debugging and warnings (use --enable-optimization to avoid this)."
#enable_debug="yes"

# Disable Boost debug mode for configure without debug/optimize options
CXXFLAGS="$CXXFLAGS -DNDEBUG"

# Handle option --enable-debug
if test x$enable_debug = xyes; then
    echo "enabling debugging compiler flags"
    if test x$enable_debug_ublas = xyes; then
       echo "enabling extra uBlas debugging compiler flags"
       CXXFLAGS="-DDEBUG -g -O2 -Wall -Werror"
    else
       CXXFLAGS="-DDEBUG -g -O2 -Wall -Werror -DNDEBUG"
    fi
fi 

# Handle option --enable-optimization
if test x$enable_optimization = xyes; then
    echo "enabling optimizing compiler flags"
    CXXFLAGS="-O3  -Wall -ansi -DNDEBUG"
fi

# Handle option --enable-debug-ublas
# FIXME: Not handled

# Add compiler/platform specific flags
case "${ax_cv_cxx_compiler_vendor}" in
   gnu) 
        # Add '-Wno-uninitialized' due to a Cygwin gcc 3.4 bug and don't use -ansi.
        # At least on Darwin, -o is not an allowed option!
        if ! test `uname -s` = Darwin && test `uname -o` = Cygwin; then
            echo "adding compiler flags for cygwin"
            CXXFLAGS="$CXXFLAGS -Wno-uninitialized"
        else
            CXXFLAGS="$CXXFLAGS -pedantic -ansi -Wno-long-long -std=c++98"
         fi
        ;;
   intel)
        echo 'WARNING: Intel C++ compiler versions <= 9.0 are incompatible with' 
        echo 'header files gcc version <= 4.0. You can specify the version of' 
        echo 'gcc header files to be used by setting the compiler option' 
        echo '-gcc-name=<name> through the environment variables'
        echo 'e.g. export INTEL_CXXFLAGS="-gcc-name=gcc-3.4"'
        CXXFLAGS="$CXXFLAGS -ansi -wd111 -wd383 -wd654 -wd810 -wd869 -wd981 -wd1125 -wd1418 -wd1419 -wd1572 -wd1599 $INTEL_CXXFLAGS"
        ;;
esac

# Add '-fPIC' for AMD 64 bit system
if test `uname -m` = x86_64; then
    echo "adding -fPIC to compiler flags"
    CXXFLAGS="$CXXFLAGS -fPIC"
fi

# Handle option --enable-pydolfin
AM_CONDITIONAL(COMPILE_PYDOLFIN, test x$enable_pydolfin = xyes)
if test x$enable_pydolfin = xyes; then
    echo "enabling compilation of PyDOLFIN"
else
    echo "disabling compilation of PyDOLFIN"
fi


# Handle option --disable-petsc
if test x$enable_petsc = xyes; then
    
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
        PETSC_CFLAGS=`make -f petsc.conf get_petsc_include PETSC_DIR=$PETSC_DIR`
        PETSC_LIBS=`make -f petsc.conf get_petsc_libs PETSC_DIR=$PETSC_DIR`
    else
        echo "*** Unable to find PETSc on your system."
        echo "*** Use --with-petsc-dir=<path> to specify the correct path for PETSc,"
        echo "*** or set the environment variable PETSC_DIR to the correct path."
        echo "*** PETSc is available at http://www-unix.mcs.anl.gov/petsc/petsc-2/"
        exit 1
    fi

    # Found PETSc, so set HAVE_PETSC_H
    CXXFLAGS="-DHAVE_PETSC_H=1 $CXXFLAGS"

    # Check for eigenvalue solver extension SLEPc. Variable SLEPC_DIR must be set.
    AC_CHECK_HEADER([$SLEPC_DIR/include/slepcversion.h], have_slepc=yes, have_slepc=no)

    if test $have_slepc = yes; then
        # Extract SLEPc variables
        SLEPC_CFLAGS=`make -f slepc.conf get_slepc_include SLEPC_DIR=$SLEPC_DIR`
        SLEPC_LIBS=`make -f slepc.conf get_slepc_libs SLEPC_DIR=$SLEPC_DIR`
        CXXFLAGS="-DHAVE_SLEPC_H=1 $CXXFLAGS"
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

# Handle option --disable-scotch
if test x$enable_scotch = xyes; then
    # Check for SCOTCH
    AC_CHECK_HEADER(scotch.h, have_scotch_header=yes, have_scotch_header=no)
    if test $have_scotch_header = yes; then
      AC_DEFINE(HAVE_SCOTCH_H,1)
      SCOTCH_LIBS="-llapack -lX11 -lm -lscotch -lscotcherrcom -lcommon"
      SCOTCH_CFLAGS="-DCOMMON_RANDOM_FIXED_SEED -DSCOTCH_RENAME"
      SCOTCH_LIBS="-lscotch -lscotcherr -lm"
    else
        echo "*** Unable to find SCOTCH on your system."
        echo "*** Use --with-scotch-dir=<path> to specify the correct path for SCOTCH,"
        echo "*** or set the environment variable SCOTCH_DIR to the correct path."
        echo "*** SCOTCH is available at http://www.labri.fr/perso/pelegrin/scotch/"
        exit 1
    fi

    # Found SCOTCH, so set HAVE_SCOTCH_H
    CXXFLAGS="-DHAVE_SCOTCH_H=1 $CXXFLAGS"

else
    # Compile without SCOTCH
    echo "disabling SCOTCH (mesh/graph partitioning will not work)"
    SCOTCH_CFLAGS=""
fi
# List of DOLFIN kernel libraries
DOLFIN_KERNEL="common elements fem function graph io la main math mesh mf nls ode pde quadrature parameter log plot"

# Generate include path for external libraries
EXTERNAL_CFLAGS="$PETSC_CFLAGS $SLEPC_CFLAGS $BOOST_CPPFLAGS $MPI_CFLAGS $XML_CPPFLAGS $UMFPACK_CFLAGS $GLIB_CFLAGS $GTS_CFLAGS $UFC_CFLAGS $SCOTCH_CFLAGS"
AC_SUBST(EXTERNAL_CFLAGS)

KERNEL_CFLAGS=""
for f in $DOLFIN_KERNEL; do
    KERNEL_INCLUDE="$KERNEL_INCLUDE -I\$(top_builddir)/src/kernel/$f"
done
AC_SUBST(KERNEL_INCLUDE)

# Generate flags for kernel
KERNEL_CFLAGS="$KERNEL_INCLUDE $EXTERNAL_CFLAGS"
AC_SUBST(KERNEL_CFLAGS)

# Generate include path for main
DOLFIN_CFLAGS="$KERNEL_INCLUDE $EXTERNAL_CFLAGS"
AC_SUBST(DOLFIN_CFLAGS)

# Extract compiler definitions
DOLFIN_DEFINITIONS=""
for i in $CXXFLAGS; do
    if test $(echo $i | sed 's/^\(..\).*/\1/') == "-D"; then
       DOLFIN_DEFINITIONS="$DOLFIN_DEFINITIONS $i"
    fi
done
AC_SUBST(DOLFIN_DEFINITIONS)

# External libraries which DOLFIN depends on
DOLFIN_EXT_LIBS="$PETSC_LIBS $SLEPC_LIBS $XML_LIBS $UMFPACK_LIBS $GLIB_LIBS $GTS_LIBS $SCOTCH_LIBS"
AC_SUBST(DOLFIN_EXT_LIBS)
echo "external libraries: $DOLFIN_EXT_LIBS"

# Library list which DOLFIN applications need to link against
DOLFIN_LIBS="-ldolfin $PETSC_LIBS"
DOLFIN_LIBS="$DOLFIN_LIBS $DOLFIN_EXT_LIBS"
AC_SUBST(DOLFIN_LIBS)

# Generate libtool library list
KERNEL_LIBOBJS=""
for f in $DOLFIN_KERNEL; do
    KERNEL_LIBOBJS="$KERNEL_LIBOBJS \$(top_builddir)/src/kernel/$f/*.lo "
done
AC_SUBST(KERNEL_LIBOBJS)

# Generate standard environment variables for installation
echo "creating dolfin.conf"
if test $prefix = NONE; then
   prefix="/usr/local"
fi
CONFIG='dolfin.conf'
rm -f $CONFIG
echo 'export PATH="'$prefix'/bin:$PATH"' >> $CONFIG
echo 'export LD_LIBRARY_PATH="'$prefix'/lib:$LD_LIBRARY_PATH"' >> $CONFIG
echo 'export PKG_CONFIG_PATH="'$prefix'/lib/pkgconfig:$PKG_CONFIG_PATH"' >> $CONFIG
echo 'export PYTHONPATH="'$prefix'/lib/python'$PYTHON_VERSION'/site-packages:$PYTHONPATH"' >> $CONFIG

# Export variables to makefiles
AC_SUBST(PACKAGE_VERSION)
AC_SUBST(PACKAGE_NAME)

# List output to be generated: Makefiles and pkgconfig configuration
AC_OUTPUT( Makefile \
           src/Makefile \
           src/kernel/Makefile \
           src/kernel/common/Makefile \
           src/kernel/common/dolfin/Makefile \
           src/kernel/elements/Makefile \
           src/kernel/elements/dolfin/Makefile \
           src/kernel/fem/Makefile \
           src/kernel/fem/dolfin/Makefile \
           src/kernel/function/Makefile \
           src/kernel/function/dolfin/Makefile \
           src/kernel/graph/Makefile \
           src/kernel/graph/dolfin/Makefile \
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
           src/kernel/mf/Makefile \
           src/kernel/mf/dolfin/Makefile \
           src/kernel/nls/Makefile \
           src/kernel/nls/dolfin/Makefile \
           src/kernel/ode/Makefile \
           src/kernel/ode/dolfin/Makefile \
           src/kernel/pde/Makefile \
           src/kernel/pde/dolfin/Makefile \
           src/kernel/plot/Makefile \
           src/kernel/plot/dolfin/Makefile \
           src/kernel/parameter/Makefile \
           src/kernel/parameter/dolfin/Makefile \
           src/kernel/quadrature/Makefile \
           src/kernel/quadrature/dolfin/Makefile \
           src/kernel/log/Makefile \
           src/kernel/log/dolfin/Makefile \
           src/lib/Makefile \
           src/pydolfin/Makefile \
           src/demo/Makefile \
           src/test/Makefile \
           src/bench/Makefile \
           src/bench/fem/Makefile \
           src/bench/la/Makefile \
           src/bench/ode/Makefile \
           src/utils/Makefile \
           src/utils/inp2dx/Makefile \
           src/utils/convert/Makefile \
           src/utils/swig/Makefile \
           src/greeting/Makefile \
           dolfin.pc )

# Display help text
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
    echo 'Warning: Installation directory is not in PATH.'
    echo ''
    echo 'To compile a program against DOLFIN (including the demos)'
    echo 'you need to update your environment variables to reflect'
    echo 'the installation directory you have chosen for DOLFIN.'
    echo 'A simple way to do this if you are using Bash-like shell'
    echo 'is to source the file dolfin.conf:'
    echo ''
    echo '  source dolfin.conf'
    echo ''
    echo 'This will update the values for the environment variables'
    echo 'PATH, LD_LIBRARY_PATH, PKG_CONFIG_PATH and PYTHONPATH.'
fi
