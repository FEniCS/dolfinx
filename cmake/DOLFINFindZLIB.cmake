set(DOLFIN_ZLIB_FOUND 0)

find_package(ZLIB)
                
if(ZLIB_FOUND)
  set(DOLFIN_ZLIB_INCLUDE_DIRS
    ${ZLIB_INCLUDE_DIRS}
    )
        
  set(DOLFIN_ZLIB_LIB_DIRS
    ${ZLIB_LIBRARY_DIRS}
    )

  set(DOLFIN_ZLIB_LIBS
    ${ZLIB_LIBRARIES}
    )

  set(DOLFIN_ZLIB_FOUND 1)
endif(ZLIB_FOUND)
