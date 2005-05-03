SUBDIRS                  = dolfin
include_HEADERS          = dolfin/dolfin_form.h \
                           dolfin/BilinearForm.h \
                           dolfin/Form.h \
                           dolfin/LinearForm.h \
                           dolfin/PDE.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LIBRARIES            = libdolfin-form.a
libdolfin_form_a_SOURCES = BilinearForm.cpp \
                           Form.cpp \
                           LinearForm.cpp \
                           PDE.cpp
