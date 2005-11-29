SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_nls.h \
                          dolfin/NewtonSolver.h \
                          dolfin/NonlinearFunction.h \
INCLUDES                = @KERNEL_CFLAGS@
lib_LIBRARIES           = libdolfin-nls.a
libdolfin_nls_a_SOURCES = NewtonSolver.cpp \
                          NonlinearFunction.cpp \
