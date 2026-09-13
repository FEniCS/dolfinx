vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO xiaoyeli/superlu_dist
  REF "v${VERSION}"
  SHA512 41afccaaffff28911504a6eff50e934a3944ef7a8613b1f9294820e45f293780b5547881e70359e7aa02e92727777d140ffc31fe53f0a36a7bedc7e977ab4849
  HEAD_REF master
)

# Upstream typo: every other source under prec-independent/ is spelled
# correctly except the MSVC-only wingetopt.c entry, which breaks the
# Windows configure (Cannot find source file: pred-independent/wingetopt.c).
vcpkg_replace_string(
  "${SOURCE_PATH}/SRC/CMakeLists.txt"
  "pred-independent/wingetopt.c"
  "prec-independent/wingetopt.c"
)

# Same class of upstream bug: every other header in this list has its
# subdirectory prefix (include/, CplusplusFactor/); the MSVC-only
# wingetopt.h entry is missing "include/", so the install step fails
# looking for it directly under SRC/.
vcpkg_replace_string(
  "${SOURCE_PATH}/SRC/CMakeLists.txt"
  "list(APPEND headers wingetopt.h)"
  "list(APPEND headers include/wingetopt.h)"
)

# unistd.h doesn't exist under MSVC. The only things util.c used it for
# (sleep()) are already commented out, so the include is unconditionally
# unused dead weight; guard it out rather than trying to shim unistd.h.
vcpkg_replace_string(
  "${SOURCE_PATH}/SRC/prec-independent/util.c"
  "#include <unistd.h>"
  "#ifndef _MSC_VER
#include <unistd.h>
#endif"
)

# Astore->nzval is void* (SuperLU_DIST's generic, precision-independent
# matrix storage type), so indexing it directly is arithmetic on a void
# pointer -- a GCC/Clang extension (treating void* like char*) that ISO C
# and MSVC both reject (C2036: 'void *': unknown size). Cast to the
# precision-specific element type each of these three files already
# otherwise assumes (matching the sizeof() used on the same line).
vcpkg_replace_string(
  "${SOURCE_PATH}/SRC/double/d3DPartition.c"
  "Astore->nzval[idx]"
  "((double *)Astore->nzval)[idx]"
)
vcpkg_replace_string(
  "${SOURCE_PATH}/SRC/single/s3DPartition.c"
  "Astore->nzval[idx]"
  "((float *)Astore->nzval)[idx]"
)
vcpkg_replace_string(
  "${SOURCE_PATH}/SRC/complex16/z3DPartition.c"
  "Astore->nzval[idx]"
  "((doublecomplex *)Astore->nzval)[idx]"
)

# SuperLU_DIST's ParMETIS TPL is satisfied by the ScotchParMETIS
# compatibility layer (scotch[metis,parmetis,ptscotch]): scotchmetisv5
# provides the METIS API used by SuperLU_DIST itself, and
# ptscotchparmetisv3 provides the ParMETIS API. TPL_PARMETIS_LIBRARIES is
# a raw (non-target) link line; scotch is built shared, so its own
# dependencies are pulled in transitively and don't need listing here.
if(VCPKG_TARGET_IS_WINDOWS)
  set(
    SUPERLU_DIST_SCOTCH_LIBRARY_NAMES
    ptscotchparmetisv3
    ptscotch
    ptscotcherr
    scotchmetisv5
    scotch
    scotcherr
  )
  # Debug/release scotch import libs share the same file names (no debug
  # postfix), but live under debug/lib and lib respectively, so the two
  # configs need distinct TPL_PARMETIS_LIBRARIES values. This must be a
  # proper CMake list (semicolon-separated): SuperLU_DIST's own CMakeLists
  # does set(PARMETIS_LIB ${TPL_PARMETIS_LIBRARIES}) unquoted, and CMake
  # only splits unquoted variable references on semicolons, never on
  # spaces. A space-joined string therefore collapses into a single list
  # element, and CMake's Ninja generator then emits that whole element as
  # one bogus combined implicit dependency, which ninja rejects with
  # "FindFirstFileExA(...): The filename ... is incorrect".
  set(SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG "")
  set(SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE "")
  foreach(
    SUPERLU_DIST_SCOTCH_LIBRARY_NAME
    IN
    LISTS SUPERLU_DIST_SCOTCH_LIBRARY_NAMES
  )
    list(
      APPEND SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG
      "${CURRENT_INSTALLED_DIR}/debug/lib/${SUPERLU_DIST_SCOTCH_LIBRARY_NAME}.lib"
    )
    list(
      APPEND SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE
      "${CURRENT_INSTALLED_DIR}/lib/${SUPERLU_DIST_SCOTCH_LIBRARY_NAME}.lib"
    )
  endforeach()
else()
  # No debug/release split needed: vcpkg's toolchain-managed library
  # search paths are already config-aware. A proper list here too, for
  # the same reason as the Windows branch above.
  set(
    SUPERLU_DIST_PARMETIS_LIBRARIES
    -lptscotchparmetisv3
    -lptscotch
    -lptscotcherr
    -lscotchmetisv5
    -lscotch
    -lscotcherr
  )
  set(
    SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG
    "${SUPERLU_DIST_PARMETIS_LIBRARIES}"
  )
  set(
    SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE
    "${SUPERLU_DIST_PARMETIS_LIBRARIES}"
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
    "-DCMAKE_C_FLAGS_INIT=-DSCOTCH_METIS_VERSION=5"
  OPTIONS_DEBUG
    "-DTPL_PARMETIS_LIBRARIES=${SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG}"
  OPTIONS_RELEASE
    "-DTPL_PARMETIS_LIBRARIES=${SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE}"
)
vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.txt")
