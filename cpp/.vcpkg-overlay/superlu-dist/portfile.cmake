vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO xiaoyeli/superlu_dist
  REF "v${VERSION}"
  SHA512 41afccaaffff28911504a6eff50e934a3944ef7a8613b1f9294820e45f293780b5547881e70359e7aa02e92727777d140ffc31fe53f0a36a7bedc7e977ab4849
  HEAD_REF master
)

# SuperLU_DIST's ParMETIS TPL is satisfied by the ScotchParMETIS
# compatibility layer (scotch[metis,parmetis,ptscotch]): scotchmetisv5
# provides the METIS API used by SuperLU_DIST itself, and
# ptscotchparmetisv3 provides the ParMETIS API. TPL_PARMETIS_LIBRARIES
# is a raw (non-target) link line, so scotch's static-library
# dependencies have to be listed out by hand on Windows, where the
# scotch port is always built static; on other platforms scotch is
# built shared, so its own dependencies are pulled in transitively.
if(VCPKG_TARGET_IS_WINDOWS)
  set(
    SUPERLU_DIST_SCOTCH_LIBRARY_NAMES
    ptscotchparmetisv3
    ptscotch
    ptscotcherr
    scotchmetisv5
    scotch
    scotcherr
    pthreadVC3
    zlib
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
else()
  set(
    SUPERLU_DIST_PARMETIS_LIBRARIES
    "-lptscotchparmetisv3 -lptscotch -lptscotcherr -lscotchmetisv5 -lscotch -lscotcherr"
  )
endif()

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DXSDK_ENABLE_Fortran=OFF
    -Denable_tests=OFF
    -Denable_examples=OFF
    -Denable_python=OFF
    -Denable_openmp=OFF
    -DBUILD_STATIC_LIBS=OFF
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
