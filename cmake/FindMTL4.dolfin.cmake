set(MTL4_FOUND 0)

message(STATUS "checking for package 'MTL4'")

find_path(MTL4_INCLUDE_DIR boost/numeric/mtl/mtl.hpp
  $ENV{MTL4_DIR}
  /usr/local/include
  /usr/include
  DOC "Directory where the MTL4 header directory is located"
  )

if(MTL4_INCLUDE_DIR)
  set(MTL4_FOUND 1)
endif(MTL4_INCLUDE_DIR)

if(MTL4_FOUND)
  message(STATUS "  found package MTL4")
  set(CMAKE_REQUIRED_INCLUDES ${MTL4_INCLUDE_DIR})
  add_definitions(-DHAS_MTL4)
else(MTL4_FOUND)
  message(STATUS "  package 'MTL4' not found")
endif(MTL4_FOUND)
