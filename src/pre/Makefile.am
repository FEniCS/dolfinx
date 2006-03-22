# Thins we need to do before building. Sort of a hack, but
# easier than figuring out how to set the dependencies with
# automake/autoconf

all:
	rm -f ../config/dolfin-config

install:
	make -C ../lib clean all
	mkdir -p $(prefix)/include/dolfin
