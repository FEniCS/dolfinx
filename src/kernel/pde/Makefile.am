SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_pde.h \
                           dolfin/LinearPDE.h \
                           dolfin/NonlinearPDE.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-pde.la
libdolfin_pde_la_SOURCES = LinearPDE.cpp \
                           NonlinearPDE.cpp
libdolfin_pde_la_LDFLAGS = -static
