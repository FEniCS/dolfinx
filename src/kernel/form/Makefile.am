SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_form.h \
                           dolfin/BLASFormData.h \
                           dolfin/BilinearForm.h \
                           dolfin/Form.h \
                           dolfin/LinearForm.h \
                           dolfin/functional.h \
                           dolfin/PDE.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LIBRARIES            = libdolfin-form.a
libdolfin_form_a_SOURCES = BLASFormData.cpp \
                           BilinearForm.cpp \
                           Form.cpp \
                           LinearForm.cpp \
                           Functional.cpp \
                           PDE.cpp
