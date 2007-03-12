if COMPILE_PYDOLFIN
PYDOLFIN = python
else
PYDOLFIN =
endif

SUBDIRS = dolfin \
          convdiff \
          elasticity \
	  elasticity-updated \
          heat \
	  navierstokes \
          plasticity \
          poisson \
          stokes \
	  ale \
	  fsi \
          lib $(PYDOLFIN)

INCLUDES = @KERNEL_CFLAGS@

nobase_include_HEADERS = dolfin/dolfin_modules.h \
                         dolfin/Solver.h
