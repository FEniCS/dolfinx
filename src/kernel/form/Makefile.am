SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_form.h \
                           dolfin/BLASFormData.h \
                           dolfin/BilinearForm.h \
                           dolfin/Form.h \
                           dolfin/LinearForm.h \
                           dolfin/Functional.h 
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES            = libdolfin-form.la
libdolfin_form_la_SOURCES = BLASFormData.cpp \
                           BilinearForm.cpp \
                           Form.cpp \
                           LinearForm.cpp \
                           Functional.cpp
libdolfin_form_la_LDFLAGS = -static
