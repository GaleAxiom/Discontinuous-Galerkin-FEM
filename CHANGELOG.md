# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CI/CD pipeline with 10+ GitHub Actions workflows
- Docker Compose configuration for easier local development
- Multi-stage Dockerfile with development and production variants
- Code coverage reporting with Codecov integration
- Static analysis workflows (cppcheck, clang-tidy)
- Dependency vulnerability scanning with Trivy
- Matrix builds testing multiple compilers and platforms
- Automated release workflow with changelog generation
- PR auto-labeling based on size and content
- Doxygen documentation generation with GitHub Pages deployment
- Build caching for faster CI/CD execution
- Issue and PR templates
- Comprehensive documentation (CI_CD.md, CONTRIBUTING.md)
- Workflow status badges in README

### Changed
- Enhanced Docker image with additional development tools
- Improved .dockerignore and .gitignore files
- Updated DOCKER.md with new features and best practices
- Enhanced README with workflow badges and better structure

### Fixed
- Docker layer caching for improved build performance

## Previous Versions

<!-- Add version history here as releases are made -->

---

## How to Update This File

When making changes:

1. Add entries under `[Unreleased]` in the appropriate category:
   - `Added` for new features
   - `Changed` for changes in existing functionality
   - `Deprecated` for soon-to-be removed features
   - `Removed` for removed features
   - `Fixed` for bug fixes
   - `Security` for vulnerability fixes

2. When releasing a new version:
   - Change `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`
   - Add new `[Unreleased]` section at the top
   - Add comparison link at the bottom

Example:
```markdown
## [1.0.0] - 2024-01-15

### Added
- Initial release
- Core DG-FEM implementation

[1.0.0]: https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/releases/tag/v1.0.0
```
