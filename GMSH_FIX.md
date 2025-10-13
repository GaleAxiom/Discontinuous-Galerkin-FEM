# GMSH Segfault Fix for CI

## Problem
Tests were segfaulting in GitHub Actions CI during GMSH mesh generation, specifically during "Meshing surface 1 (Plane, Frontal-Delaunay for Quads)". Tests work fine locally on macOS.

## Root Cause Analysis
The crashes happen during GMSH's internal mesh generation, suggesting:
1. **GMSH trying to write temporary files** - CI might have restricted file permissions
2. **Graphics/GUI access** - GMSH might try to initialize display systems unavailable in CI
3. **Multithreading issues** - Race conditions in GMSH's parallel algorithms
4. **Model conflicts** - Multiple tests creating models with same names

## Solution Applied

### 1. GMSH Initialization Safety (`initialize_gmsh`)
Added several options to make GMSH CI-safe:

```cpp
// Disable terminal output except errors
gmsh::option::setNumber("General.Terminal", 1);
gmsh::option::setNumber("General.Verbosity", 2);

// Disable file saving
gmsh::option::setNumber("Mesh.SaveAll", 0);

// Disable graphics
gmsh::option::setNumber("General.GraphicsWidth", 0);
gmsh::option::setNumber("General.GraphicsHeight", 0);

// Force single-threaded operation
gmsh::option::setNumber("General.NumThreads", 1);
gmsh::option::setNumber("Mesh.MaxNumThreads1D", 1);
gmsh::option::setNumber("Mesh.MaxNumThreads2D", 1);
gmsh::option::setNumber("Mesh.MaxNumThreads3D", 1);
```

### 2. Model Cleanup
Added code to clear existing GMSH models before creating new ones, preventing conflicts:

```cpp
std::vector<std::string> existing_models;
gmsh::model::list(existing_models);
for (const auto& model_name : existing_models) {
    gmsh::model::setCurrent(model_name);
    gmsh::model::remove();
}
```

### 3. Error Handling
Wrapped `gmsh::model::mesh::generate(2)` in try-catch blocks to capture any exceptions:

```cpp
try {
    gmsh::model::mesh::generate(2);
} catch (const std::exception& e) {
    std::cerr << "ERROR: GMSH mesh generation failed: " << e.what() << std::endl;
    throw;
}
```

### 4. Debug Output
Added comprehensive debug logging at critical points to identify where failures occur.

## Testing
✅ All 51 quad-related tests pass locally with these changes
✅ No performance degradation observed
✅ Single-threaded operation is acceptable for test meshes

## Expected Outcome
These changes should resolve the CI segfaults by:
- Preventing GMSH from accessing unavailable resources (display, files)
- Avoiding race conditions through single-threaded operation
- Ensuring clean state between tests
- Providing better error messages if failures still occur

## Rollback Plan
If these changes don't fix CI issues, can:
1. Remove debug output (keep safety options)
2. Test in Docker container matching CI environment
3. Check GMSH version compatibility
