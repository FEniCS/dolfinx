# Compiles a CUDA translation unit that includes the DOLFINx headers with
# nvcc, against an installed (host-only) DOLFINx.
cmake_minimum_required(VERSION 3.26)

project(dolfinx-nvcc LANGUAGES CXX CUDA)

find_package(DOLFINX REQUIRED)

add_executable(nvcc-check main.cu)
target_link_libraries(nvcc-check dolfinx)
set_target_properties(
  nvcc-check
  PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
)
