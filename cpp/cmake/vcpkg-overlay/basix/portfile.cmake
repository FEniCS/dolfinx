# This port builds Basix from an already-checked-out local source
# tree instead of downloading a tagged release: dolfinx CI tests
# against Basix's current branch tip (or a specific feature branch),
# so there is no fixed commit to pin/hash here. The workflow that
# invokes vcpkg for this port must set the BASIX_SOURCE_DIR
# environment variable to that checked-out cpp/ directory, and must
# also keep vcpkg.json's version-string in step with the checked-out
# commit, so vcpkg's binary cache doesn't reuse a build made from a
# different commit under the same version.
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
