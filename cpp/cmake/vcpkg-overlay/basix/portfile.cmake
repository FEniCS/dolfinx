# This port builds Basix from an already-checked-out local source tree
# instead of a pinned release, since dolfinx CI tracks Basix's current
# branch tip.
if(NOT DEFINED ENV{BASIX_SOURCE_DIR})
  message(
    FATAL_ERROR
    "The basix overlay port requires the BASIX_SOURCE_DIR environment "
    "variable to point at a checked-out Basix cpp/ source directory."
  )
endif()
set(SOURCE_PATH "$ENV{BASIX_SOURCE_DIR}")

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DBUILD_SHARED_LIBS=ON
    -DINSTALL_RUNTIME_DEPENDENCIES=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME Basix CONFIG_PATH lib/cmake/basix)
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/../LICENSE")
