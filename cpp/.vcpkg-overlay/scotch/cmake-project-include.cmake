# Scotch sets no WINDOWS_EXPORT_ALL_SYMBOLS/dllexport annotations itself,
# so DLLs would export nothing. Its public API is function-only (no
# exported data symbols), so auto-exporting everything is sufficient.
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
