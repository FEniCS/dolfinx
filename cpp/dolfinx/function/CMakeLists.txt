set(HEADERS_function
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_function.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Constant.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Function.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionSpace.h
  PARENT_SCOPE)

target_sources(dolfinx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Function.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionSpace.cpp
)
