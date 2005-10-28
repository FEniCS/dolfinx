SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_nls.h \
                          dolfin/NonlinearFunctional.h \
                          dolfin/NonlinearSolver.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LIBRARIES           = libdolfin-nls.a
libdolfin_nls_a_SOURCES = NonlinearFunctional.cpp \
                          NonlinearSolver.cpp 
