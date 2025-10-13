# Code Formatting Guide

This project uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to maintain consistent code style across all C++ source files.

## Installation

### macOS (Homebrew)

```bash
brew install llvm
```

### Ubuntu/Debian

```bash
sudo apt-get install clang-format
```

### Other Systems

Download from [LLVM releases](https://releases.llvm.org/) or install via your package manager.

## Usage

### Format All Files

Run the format script from the project root:

```bash
./format-code.sh
```

This will automatically format all `.cpp` and `.hpp` files in the `dgfem/` directory.

### Format Specific File

```bash
clang-format -i path/to/file.cpp
```

### Check Formatting (without modifying)

```bash
clang-format --dry-run --Werror path/to/file.cpp
```

## Pre-commit Hook (Optional)

To automatically check formatting before each commit:

```bash
ln -s ../../hooks/pre-commit .git/hooks/pre-commit
```

This will prevent commits with incorrectly formatted code.

## Code Style Guidelines

The `.clang-format` configuration enforces the following style:

### Basic Style

- **Base Style**: LLVM
- **C++ Standard**: C++17
- **Indentation**: 4 spaces (no tabs)
- **Column Limit**: 100 characters
- **Brace Style**: Attached (K&R style)

### Naming Conventions

- **Classes/Structs**: `PascalCase` (e.g., `DGMesh`, `BoundaryCondition`)
- **Functions/Methods**: `snake_case` (e.g., `compute_element_data`, `get_element_neighbors`)
- **Variables**: `snake_case` with trailing underscore for members (e.g., `n_elements_`, `dg_space_`)
- **Constants**: `SCREAMING_SNAKE_CASE` or `kPascalCase`

### Pointers and References

Left-aligned:

```cpp
Type* pointer;
Type& reference;
const Type* const_pointer;
```

### Function Declarations

```cpp
// Short functions on single line (if they fit)
int get_value() const { return value_; }

// Multi-line parameter lists
void long_function_name(
    const Eigen::MatrixXd& matrix,
    const std::vector<int>& indices,
    double tolerance);
```

### Constructor Initializer Lists

```cpp
DGMesh::DGMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& elements)
    : vertices_(vertices), 
      elements_(elements),
      n_elements_(elements.rows()) {
    // Constructor body
}
```

### Include Order

1. Project headers (`dgfem/...`)
2. Third-party headers (Eigen, gmsh)
3. C++ standard library headers
4. C standard library headers

```cpp
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include <Eigen/Dense>
#include <gmsh.h>
#include <iostream>
#include <vector>
#include <cmath>
```

### Comments

```cpp
/**
 * @brief Brief description of function
 * @param param1 Description of parameter
 * @return Description of return value
 */
void function(int param1);

// Single-line comments for implementation details
int value = compute();  // Trailing comment
```

### Spacing

```cpp
// Spaces around operators
int result = a + b * c;

// No spaces inside parentheses
function(arg1, arg2);

// Spaces after control statements
if (condition) {
    do_something();
}

for (int i = 0; i < n; ++i) {
    process(i);
}
```

## Editor Integration

### VS Code

Install the [Clang-Format extension](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format):

```json
{
    "editor.formatOnSave": true,
    "clang-format.executable": "/opt/homebrew/opt/llvm/bin/clang-format"
}
```

### CLion/IntelliJ

1. Go to **Settings → Editor → Code Style → C/C++**
2. Click **Set from...** → **Predefined Style** → **LLVM**
3. Or import the `.clang-format` file directly

### Vim/Neovim

Add to your config:

```vim
" Format on save
autocmd BufWritePre *.cpp,*.hpp silent! !clang-format -i %
```

### Emacs

Add to your config:

```elisp
(require 'clang-format)
(add-hook 'c++-mode-hook
          (lambda () (add-hook 'before-save-hook 'clang-format-buffer nil 'local)))
```

## CI/CD

The GitHub Actions workflow automatically checks code formatting on every pull request. Files that don't match the style will cause the CI to fail.

To ensure your PR passes:

1. Run `./format-code.sh` before committing
2. Or install the pre-commit hook: `ln -s ../../hooks/pre-commit .git/hooks/pre-commit`

## Troubleshooting

### "clang-format not found"

Make sure clang-format is in your PATH, or the script will look for it in:
- `/opt/homebrew/opt/llvm/bin/clang-format` (Homebrew Apple Silicon)
- `/usr/local/opt/llvm/bin/clang-format` (Homebrew Intel)
- System PATH

### Different formatting between local and CI

Make sure you're using a compatible version of clang-format (14+). Check with:

```bash
clang-format --version
```

The CI uses the version from Ubuntu's package repository.

### Files not being formatted

Ensure files are in the correct directories:
- `dgfem/src/`
- `dgfem/include/`
- `dgfem/tests/`
- `dgfem/examples/`

Files in `build/`, `external/`, or other directories are intentionally excluded.
