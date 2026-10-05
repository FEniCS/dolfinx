target_sources(dolfinx PRIVATE SNESSolver.cpp)

target_sources(dolfinx PUBLIC FILE_SET HEADERS FILES SNESSolver.h dolfinx_nls.h)
