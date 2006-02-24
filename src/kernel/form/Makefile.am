SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_form.h \
                           dolfin/BLASFormData.h \
                           dolfin/BilinearForm.h \
                           dolfin/Form.h \
                           dolfin/GenericPDE.h \
                           dolfin/LinearForm.h \
                           dolfin/LinearPDE.h \
                           dolfin/NonlinearPDE.h \
                           dolfin/Functional.h \
                           dolfin/PDE.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES            = libdolfin-form.la
libdolfin_form_la_SOURCES = BLASFormData.cpp \
                           BilinearForm.cpp \
                           Form.cpp \
                           GenericPDE.cpp \
                           LinearForm.cpp \
                           LinearPDE.cpp \
                           NonlinearPDE.cpp \
                           Functional.cpp \
                           PDE.cpp
libdolfin_form_la_LDFLAGS = -static
