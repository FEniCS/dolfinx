dnl @synopsis AC_PYTHON_DEVEL
dnl
dnl Checks for Python and tries to get the include path to 'Python.h'.
dnl It provides the $(PYTHON_CPPFLAGS) and $(PYTHON_LDFLAGS) output
dnl variable.
dnl
dnl @category InstalledPackages
dnl @author Sebastian Huber <sebastian-huber@web.de>
dnl @author Alan W. Irwin <irwin@beluga.phys.uvic.ca>
dnl @author Rafael Laboissiere <laboissiere@psy.mpg.de>
dnl @author Andrew Collier <colliera@nu.ac.za>
dnl @version 2004-07-14
dnl @license GPLWithACException

AC_DEFUN([AC_PYTHON_DEVEL],[
	#
	# should allow for checking of python version here...
	#
	AC_REQUIRE([AM_PATH_PYTHON])

	# Check for Python include path
	AC_MSG_CHECKING([for Python include path])
   python_path=`$PYTHON -c 'from distutils import sysconfig; print sysconfig.get_python_inc();'`
   f_python_path=`find $python_path -type f -name Python.h -print | sed "1q"`
   if test -n "$f_python_path" ; then
      python_path=`dirname $f_python_path`
   fi
	AC_MSG_RESULT([$python_path])
	if test -z "$python_path" ; then
		AC_MSG_ERROR([cannot find Python include path])
	fi
	AC_SUBST([PYTHON_CPPFLAGS],[-I$python_path])

	# Check for Python library path - python exec_prefix changed above, need to
	# reset
	AC_MSG_CHECKING([for Python library path])
   python_path=`$PYTHON -c 'import sys; print sys.exec_prefix;'`
	for i in "$python_path/lib/python$PYTHON_VERSION/config/" "$python_path/lib/python$PYTHON_VERSION/" "$python_path/lib/python/config/" "$python_path/lib/python/" "$python_path/" ; do
		python_path=`find $i -type f -name libpython$PYTHON_VERSION.* -print | sed "1q"`
		if test -n "$python_path" ; then
			break
		fi
	done
	python_path=`echo $python_path | sed "s,/libpython.*$,,"`
	AC_MSG_RESULT([$python_path])
	if test -z "$python_path" ; then
		AC_MSG_ERROR([cannot find Python library path])
	fi
	AC_SUBST([PYTHON_LDFLAGS],["-L$python_path -lpython$PYTHON_VERSION"])
	#
	python_site=`echo $python_path | sed "s/config/site-packages/"`
	AC_SUBST([PYTHON_SITE_PKG],[$python_site])
	#
	# libraries which must be linked in when embedding
	#
	AC_MSG_CHECKING(python extra libraries)
	PYTHON_EXTRA_LIBS=`$PYTHON -c "import distutils.sysconfig; \
                conf = distutils.sysconfig.get_config_var; \
                print conf('LOCALMODLIBS')+' '+conf('LIBS')"
	AC_MSG_RESULT($PYTHON_EXTRA_LIBS)`
	AC_SUBST(PYTHON_EXTRA_LIBS)
])
