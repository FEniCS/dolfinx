SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_pde.h \
                           dolfin/GenericPDE.h \
                           dolfin/LinearPDE.h \
                           dolfin/NonlinearPDE.h \
                           dolfin/PDE.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-pde.la
libdolfin_pde_la_SOURCES = GenericPDE.cpp \
                           LinearPDE.cpp \
                           NonlinearPDE.cpp \
                           PDE.cpp
libdolfin_pde_la_LDFLAGS = -static
