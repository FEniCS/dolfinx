pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = dolfin.pc

SUBDIRS = src

demo:
	make -C src/demo/

modules:
	make -C src/modules

modules_install:
	make -C src/modules install
