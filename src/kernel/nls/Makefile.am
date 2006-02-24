SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_nls.h \
                          dolfin/NewtonSolver.h \
                          dolfin/NonlinearProblem.h 
INCLUDES                = @KERNEL_CFLAGS@
lib_LTLIBRARIES           = libdolfin-nls.la
libdolfin_nls_la_SOURCES = NewtonSolver.cpp \
                          NonlinearProblem.cpp
libdolfin_nls_la_LDFLAGS = -static
