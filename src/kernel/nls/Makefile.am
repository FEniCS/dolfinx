SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_nls.h \
                          dolfin/NewtonSolver.h \
                          dolfin/NonlinearFunction.h \
                          dolfin/NonlinearPDE.h 
INCLUDES                = @KERNEL_CFLAGS@
lib_LTLIBRARIES           = libdolfin-nls.la
libdolfin_nls_la_SOURCES = NewtonSolver.cpp \
                          NonlinearFunction.cpp \
                          NonlinearPDE.cpp 
libdolfin_nls_la_LDFLAGS = -static
