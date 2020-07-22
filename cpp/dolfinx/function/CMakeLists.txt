set(HEADERS_function
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_function.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Constant.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Expression.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Function.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionSpace.h
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  PARENT_SCOPE)

target_sources(dolfinx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionSpace.cpp
)
