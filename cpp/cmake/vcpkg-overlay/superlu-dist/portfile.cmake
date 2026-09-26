vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO jhale/superlu_dist
  REF 626c8e5a9c8049d2ea18baa683742c45e908402b
  SHA512 7595fbd347e7781830da68b02c7c08195f095743c8d079cee9eceda3af3466ca67b99f4e341b8cc1868d02f0a6941ff39ee13e25ac25ae770ed0add681d86c53
  HEAD_REF jhale/windows-fixes
)

# SuperLU_DIST's ParMETIS TPL is satisfied by the ScotchParMETIS
# compatibility layer (scotch[metis,parmetis,ptscotch]): scotchmetisv5
# provides the METIS API used by SuperLU_DIST itself, and
# ptscotchparmetisv3 provides the ParMETIS API. TPL_PARMETIS_LIBRARIES
# is a raw (non-target) link line, so scotch's static-library
# dependencies have to be listed out by hand, since the scotch port
# is always built static.
set(
  SUPERLU_DIST_SCOTCH_LIBRARY_NAMES
  ptscotchparmetisv3
  ptscotch
  ptscotcherr
  scotchmetisv5
  scotch
  scotcherr
  z
  bz2
  lzma
)
set(SUPERLU_DIST_PARMETIS_LIBRARIES "")
foreach(
  SUPERLU_DIST_SCOTCH_LIBRARY_NAME
  IN
  LISTS SUPERLU_DIST_SCOTCH_LIBRARY_NAMES
)
  string(
    APPEND SUPERLU_DIST_PARMETIS_LIBRARIES
    "${CURRENT_INSTALLED_DIR}/lib/${SUPERLU_DIST_SCOTCH_LIBRARY_NAME}.lib "
  )
endforeach()

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DXSDK_ENABLE_Fortran=OFF
    -Denable_tests=OFF
    -Denable_examples=OFF
    -Denable_python=OFF
    -Denable_openmp=OFF
    -DBUILD_STATIC_LIBS=ON
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF
    -DTPL_ENABLE_LAPACKLIB=ON
    -DTPL_ENABLE_PARMETISLIB=ON
    "-DTPL_PARMETIS_INCLUDE_DIRS=${CURRENT_INSTALLED_DIR}/include"
    "-DTPL_PARMETIS_LIBRARIES=${SUPERLU_DIST_PARMETIS_LIBRARIES}"
    "-DCMAKE_C_FLAGS_INIT=-DSCOTCH_METIS_VERSION=5"
)
vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.txt")
