# Init autoconf
AC_INIT(dolfin, 0.4.2, dolfin@math.chalmers.se)

# Init automake
AM_INIT_AUTOMAKE($PACKAGE, $VERSION, no-define)

# Tell automake not to generate Makefile.in unless asked to.
# This way the code can be installed on systems without automake installed.
AM_MAINTAINER_MODE

# Option --enable-debug
AC_ARG_ENABLE(debug,
              AC_HELP_STRING([--enable-debug],
                             [Turn on debugging and warnings]),
              CXXFLAGS='-DDEBUG=1 -g -O2 -Wall -Werror')

# Option --disable-curses
AC_ARG_ENABLE(curses,
              AC_HELP_STRING([--disable-curses],
                             [Compile without curses]),
              CXXFLAGS=$CXXFLAGS' -DNO_CURSES=1')

# Checks for programs
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_LN_S

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T

# Check for libxml2
AC_CHECK_PROG(libxml2_found, xml2-config, yes, no)
if test $libxml2_found = yes; then	  
	XML2_INCLUDES=`xml2-config --cflags`
	XML2_LFLAGS=`xml2-config --libs`
 	AC_SUBST(XML2_INCLUDES)
	AC_SUBST(XML2_LFLAGS)
else
 	echo "*** Unable to find libxml2 development files on your system."
        echo "*** Perhaps you need to install the package libxml2-dev?"
	exit 1
fi

# Check for curses
AC_SEARCH_LIBS(wbkgdset, ncurses,,echo "*** Unable to find ncurses on your system."; exit 1)

# Note: need to be in the correct order, and some are needed twice.
# Not very pretty... Maybe someone knows how to fix this?
#DOLFIN_KERNEL="ode io mesh fem element mesh io fem form quadrature map la math common settings log settings common main ode"
DOLFIN_KERNEL="common element fem form function io la log main map math mesh ode quadrature settings"
DOLFIN_KERNEL="$DOLFIN_KERNEL $DOLFIN_KERNEL $DOLFIN_KERNEL $DOLFIN_KERNEL"

# Find modules
cd src/modules
./scanmodules.sh
cd ../..
DOLFIN_MODULES=""
for l in `cat src/modules/modules.list | grep -v "#"`; do
	DOLFIN_MODULES="$DOLFIN_MODULES$l "
done

# Generate include path for modules (including kernel code)
MODULE_INCLUDES=""
DOLFIN_MAIN="main"
DIRS="$DOLFIN_MAIN $DOLFIN_KERNEL"
for f in $DIRS; do
	MODULE_INCLUDES="$MODULE_INCLUDES-I\$(top_builddir)/src/kernel/$f "
done
AC_SUBST(MODULE_INCLUDES)

# Generate include path for main (including also module code)
MAIN_INCLUDES=""
DIRS="$DOLFIN_MODULES"
for f in $DIRS; do
	MAIN_INCLUDES="$MAIN_INCLUDES-I\$(top_builddir)/src/modules/$f "
done
MAIN_INCLUDES="$MODULE_INCLUDES $MAIN_INCLUDES"
AC_SUBST(MAIN_INCLUDES)

# Generate library list for linking in the correct order.
DOLFIN_LFLAGS=""
DIRS="$DOLFIN_MAIN $DOLFIN_MODULES $DOLFIN_KERNEL"
for f in $DIRS; do
	DOLFIN_LFLAGS="$DOLFIN_LFLAGS-ldolfin-$f "
done
AC_SUBST(DOLFIN_LFLAGS)

# Seems to be needed for CXXFLAGS to get to src/config
AC_SUBST(CXXFLAGS)

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
           src/modules/template/Makefile \
           src/modules/poisson/Makefile \
           src/modules/convdiff/Makefile \
           src/modules/heat/Makefile \
           src/modules/navierstokes/Makefile \
           src/modules/odesolver/Makefile \
           src/config/Makefile \
           src/post/Makefile \
           src/demo/Makefile \
           src/demo/la/Makefile \
           src/demo/solvers/Makefile \
           src/demo/solvers/ode/Makefile \
           src/demo/solvers/ode/stiff/Makefile \
           src/demo/solvers/ode/mechanical/Makefile \
           src/utils/Makefile \
           src/utils/inp2dx/Makefile \
           src/greeting/Makefile )
