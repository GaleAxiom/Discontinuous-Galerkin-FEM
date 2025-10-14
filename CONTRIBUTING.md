# Contributing to DG-FEM

Thank you for considering contributing to the Discontinuous Galerkin Finite Element Method project! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:
- Familiarity with C++20
- Basic understanding of finite element methods (helpful but not required)
- Git and GitHub account
- Development environment set up (see [README.md](README.md))

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/Discontinuous-Galerkin-FEM.git
   cd Discontinuous-Galerkin-FEM
   ```

2. **Set up the development environment**
   
   Using Docker (recommended):
   ```bash
   docker-compose up dev
   ```
   
   Or install dependencies locally (see [README.md](README.md#prerequisites))

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Code Style

### Formatting

This project uses **clang-format** for consistent code formatting.

**Before committing, always run:**
```bash
./format-code.sh
```

**Key style guidelines:**
- **Indentation:** 4 spaces (no tabs)
- **Line length:** 100 characters maximum
- **Pointer/Reference:** Left-aligned (`Type* ptr`, `Type& ref`)
- **Braces:** LLVM style (opening brace on same line)
- **Naming conventions:**
  - Variables: `snake_case`
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

See [CODE_STYLE.md](CODE_STYLE.md) for complete style guidelines.

### Code Quality

- Write clear, self-documenting code
- Add comments for complex algorithms or non-obvious logic
- Use meaningful variable and function names
- Keep functions small and focused
- Follow RAII principles for resource management
- Use `const` and `constexpr` where appropriate

## 🧪 Testing

### Writing Tests

- Add tests for all new functionality
- Use Google Test framework
- Place tests in `dgfem/tests/`
- Follow existing test structure and naming conventions

**Test file naming:**
```
dgfem/tests/category/test_feature.cpp
```

**Test naming:**
```cpp
TEST(TestSuiteName, TestCaseName) {
    // Test implementation
}
```

### Running Tests

```bash
# Build and run all tests
cmake -B build -S dgfem
cmake --build build
cd build && ctest --output-on-failure

# Or using Docker
./build-docker.sh
```

## 📚 Documentation

### Code Documentation

- Use Doxygen-style comments for public APIs
- Document function parameters and return values
- Provide usage examples for complex features

**Example:**
```cpp
/**
 * @brief Compute the numerical flux at element interfaces
 * 
 * @param u_left Left state vector
 * @param u_right Right state vector
 * @param normal Outward normal vector
 * @return Numerical flux vector
 */
Eigen::VectorXd compute_flux(
    const Eigen::VectorXd& u_left,
    const Eigen::VectorXd& u_right,
    const Eigen::Vector2d& normal
);
```

### Documentation Files

- Update relevant `.md` files when adding features
- Keep README.md up to date
- Document breaking changes clearly

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure your code builds**
   ```bash
   cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   ```

2. **Run all tests**
   ```bash
   cd build && ctest --output-on-failure
   ```

3. **Format your code**
   ```bash
   ./format-code.sh
   ```

4. **Check for warnings**
   ```bash
   cmake --build build 2>&1 | grep -i warning
   ```

### Submitting a Pull Request

1. **Push your changes**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request on GitHub**
   - Use the PR template provided
   - Write a clear title and description
   - Link related issues
   - Add screenshots for UI/visual changes

3. **Wait for CI/CD checks**
   - All workflows must pass
   - Address any failures before review

4. **Respond to review feedback**
   - Make requested changes
   - Re-request review after updates

### PR Guidelines

- **Keep PRs focused:** One feature or fix per PR
- **Keep PRs small:** Aim for < 500 lines changed when possible
- **Write clear commit messages:** Describe what and why, not how
- **Update tests:** Add/modify tests for your changes
- **Update documentation:** If behavior changes, update docs

## 🐛 Reporting Bugs

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Test with latest version** to see if it's already fixed
3. **Try to reproduce** with minimal example

### Bug Report Checklist

When reporting a bug, include:
- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (OS, compiler, versions)
- **Error messages** or logs (use code blocks)
- **Minimal code example** if applicable

Use the bug report template provided when creating an issue.

## 💡 Suggesting Features

We welcome feature suggestions! When proposing a feature:

1. **Check existing issues/PRs** to avoid duplicates
2. **Describe the problem** you're trying to solve
3. **Propose a solution** with rationale
4. **Consider alternatives** and trade-offs
5. **Indicate willingness to implement** if applicable

Use the feature request template provided when creating an issue.

## 🏗️ Project Structure

Understanding the codebase organization:

```
dgfem/
├── include/dgfem/    # Public headers
│   ├── core/         # Core classes (Mesh, Element, Space)
│   ├── solver/       # Solver implementations
│   ├── basis/        # Basis functions
│   ├── boundary/     # Boundary conditions
│   └── utils/        # Utilities
├── src/              # Implementation files
├── tests/            # Unit tests
└── examples/         # Example programs
```

## 🔧 CI/CD Pipeline

The project uses GitHub Actions for continuous integration. See [CI_CD.md](CI_CD.md) for details.

**Key workflows:**
- **Build & Test:** Runs on every push/PR
- **Matrix Build:** Tests multiple compilers/platforms
- **Code Coverage:** Tracks test coverage
- **Static Analysis:** Catches potential bugs
- **Format Check:** Ensures code style compliance

All checks must pass before a PR can be merged.

## 📋 Commit Message Guidelines

Write clear, descriptive commit messages:

**Format:**
```
<type>: <short summary>

<optional detailed description>

<optional footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tooling changes
- `perf`: Performance improvements

**Example:**
```
feat: add high-order Dubiner basis functions

Implement Dubiner orthogonal basis functions for triangular elements
up to order 10. Includes automatic quadrature rule selection.

Closes #42
```

## 🤝 Code Review Process

### For Authors

- Respond to feedback promptly
- Ask questions if feedback is unclear
- Make requested changes or discuss alternatives
- Mark conversations as resolved after addressing

### For Reviewers

- Be constructive and specific
- Focus on code, not the person
- Suggest improvements, don't just criticize
- Approve when satisfied with changes

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## 🙋 Getting Help

- **Documentation:** Check [README.md](README.md), [DOCKER.md](DOCKER.md), [CI_CD.md](CI_CD.md)
- **Issues:** Search or create an issue
- **Discussions:** Use GitHub Discussions for questions

## 🎉 Recognition

Contributors are recognized in:
- Git commit history
- Release notes for significant contributions
- Special thanks in documentation

Thank you for contributing to DG-FEM! 🚀
