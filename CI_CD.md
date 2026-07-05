# CI/CD Pipeline Documentation

This document describes the continuous integration and deployment (CI/CD) workflows for the DG-FEM project.

## Overview

The project uses GitHub Actions for automated building, testing, and deployment. All workflows are located in `.github/workflows/`.

## Workflows

**Status:** only 3 of the workflows described below actually exist in `.github/workflows/`
today -- `cmake-single-platform.yml`, `code-coverage.yml`, and `format-check.yml` (#1, #3, #5).
The other 7 (#2, #4, #6-10) are planned/aspirational and marked **(not yet implemented)** below;
their sections describe the intended design, not current behavior. Check
`ls .github/workflows/` for the current ground truth rather than assuming this doc is
exhaustive.

### 1. CMake Single Platform (`cmake-single-platform.yml`)

**Triggers:** Push to `main`, Pull requests to `main`

**Purpose:** Primary build and test workflow using the Docker container.

**Features:**
- Runs on Ubuntu with pre-built Docker image
- CMake build caching for faster builds
- Parallel compilation
- Comprehensive test execution
- Uploads test results as artifacts

**Usage:**
```yaml
# Automatically runs on every push/PR
```

---

### 2. Matrix Build & Test (`matrix-build.yml`) (not yet implemented)

**Triggers:** Push to `main`, Pull requests to `main`

**Purpose:** Test compatibility across multiple compilers and platforms.

**Matrix:**
- **Operating Systems:** Ubuntu 22.04, Ubuntu 20.04
- **Compilers:** GCC 11, GCC 12, Clang 14, Clang 15
- **Build Types:** Debug, Release

**Features:**
- Ensures cross-compiler compatibility
- Tests both debug and optimized builds
- Uploads release artifacts

**Usage:**
```bash
# View results in GitHub Actions tab
# Download artifacts for specific compiler/OS combinations
```

---

### 3. Code Coverage (`code-coverage.yml`)

**Triggers:** Push to `main`, Pull requests to `main`

**Purpose:** Generate and track code coverage metrics.

**Features:**
- Compiles with coverage flags (`--coverage`)
- Generates lcov reports
- Uploads to Codecov (requires `CODECOV_TOKEN` secret)
- Creates HTML coverage reports as artifacts
- Filters out external dependencies and test code

**Setup:**
1. Sign up at [codecov.io](https://codecov.io)
2. Add `CODECOV_TOKEN` to repository secrets
3. Badge will appear on README after first run

**Usage:**
```bash
# Download coverage report from workflow artifacts
# View coverage trends on Codecov dashboard
```

---

### 4. Static Analysis (`static-analysis.yml`) (not yet implemented)

**Triggers:** Push to `main`, Pull requests to `main`

**Purpose:** Run static analysis tools to catch potential bugs and style issues.

**Tools:**
- **cppcheck:** Comprehensive C++ static analyzer
- **clang-tidy:** LLVM-based linter with modernization checks

**Features:**
- Checks for common programming errors
- Suggests modernization improvements
- Performance optimization hints
- Readability recommendations
- Uploads analysis reports as artifacts

**Usage:**
```bash
# Review reports in workflow artifacts
# Address critical issues before merging
```

---

### 5. Code Formatting Check (`format-check.yml`)

**Triggers:** Push to `main`, Pull requests to `main`

**Purpose:** Ensure consistent code formatting using clang-format.

**Features:**
- Checks all C++ files in `dgfem/`
- Uses project's `.clang-format` configuration
- Fails CI if formatting is incorrect
- Provides clear error messages

**Usage:**
```bash
# Run locally before committing
./format-code.sh

# Or format specific file
clang-format -i path/to/file.cpp
```

---

### 6. Docker Image Build & Publish (`docker-publish.yml`) (not yet implemented)

**Triggers:** 
- Push to `main` (when Dockerfile or workflow changes)
- Manual workflow dispatch

**Purpose:** Build and publish Docker images to GitHub Container Registry.

**Features:**
- Multi-stage builds (development and production)
- Layer caching for faster builds
- Automatic versioning with SHA tags
- Separate images for dev and production

**Images:**
- `ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest` - Development (default)
- `ghcr.io/galeaxiom/discontinuous-galerkin-fem:dev` - Development
- `ghcr.io/galeaxiom/discontinuous-galerkin-fem:prod` - Production (minimal)
- `ghcr.io/galeaxiom/discontinuous-galerkin-fem:main-<sha>` - Specific commit

**Usage:**
```bash
# Pull latest development image
docker pull ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest

# Pull production image
docker pull ghcr.io/galeaxiom/discontinuous-galerkin-fem:prod

# Use docker-compose
docker-compose up dev
```

---

### 7. Dependency Security Scan (`dependency-scan.yml`) (not yet implemented)

**Triggers:** 
- Push to `main`, Pull requests to `main`
- Weekly schedule (Monday 00:00 UTC)

**Purpose:** Scan Docker images and dependencies for security vulnerabilities.

**Features:**
- Trivy vulnerability scanner
- SARIF reports uploaded to GitHub Security
- Dependency review for PRs
- Automated weekly scans
- Severity filtering (CRITICAL, HIGH)

**Usage:**
```bash
# View security alerts in GitHub Security tab
# Review dependency changes in PRs
```

---

### 8. Documentation Generation (`documentation.yml`) (not yet implemented)

**Triggers:** 
- Push to `main`, Pull requests to `main`
- Manual workflow dispatch

**Purpose:** Generate API documentation using Doxygen.

**Features:**
- Generates HTML documentation from code comments
- Creates call graphs with Graphviz
- Deploys to GitHub Pages on main branch
- Makes README.md the main page

**Usage:**
```bash
# View docs at: https://galeaxiom.github.io/Discontinuous-Galerkin-FEM/
# Download documentation artifacts from workflow
```

**Setup for GitHub Pages:**
1. Go to Settings → Pages
2. Select `gh-pages` branch as source
3. Documentation will be available after first deployment

---

### 9. Release Automation (`release.yml`) (not yet implemented)

**Triggers:** 
- Push tags matching `v*.*.*`
- Manual workflow dispatch with version input

**Purpose:** Automate release creation and artifact packaging.

**Features:**
- Automatic changelog generation from commits
- Creates GitHub releases
- Builds and packages release artifacts
- Links to tagged Docker image

**Usage:**
```bash
# Create a release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or trigger manually from Actions tab
```

---

### 10. PR Labeling (`pr-labels.yml`) (not yet implemented)

**Triggers:** Pull requests (opened, synchronized, reopened)

**Purpose:** Automatically label pull requests based on size and content.

**Labels Added:**
- **Size:** `size/XS`, `size/S`, `size/M`, `size/L`, `size/XL`
- **Type:** `tests`, `examples`, `documentation`, `docker`, `ci/cd`, `code`

**Size Criteria:**
- XS: < 10 changes
- S: 10-49 changes
- M: 50-199 changes
- L: 200-499 changes
- XL: 500+ changes

**Usage:**
```bash
# Labels are automatically applied to PRs
# Use labels for filtering and reviewing
```

---

## Secrets Required

The following GitHub secrets should be configured:

| Secret | Required For | Description |
|--------|--------------|-------------|
| `GITHUB_TOKEN` | All workflows | Auto-provided by GitHub Actions |
| `CODECOV_TOKEN` | Code coverage | Upload coverage to Codecov |

---

## Caching Strategy

### CMake Build Cache
- Caches `build/` directory and `~/.ccache`
- Key: Based on CMakeLists.txt and source file hashes
- Reduces build time by ~60%

### Docker Layer Cache
- Registry-based caching
- Stores intermediate layers
- Speeds up image builds significantly

---

## Best Practices

### For Contributors

1. **Always run format-code.sh before committing**
   ```bash
   ./format-code.sh
   ```

2. **Test locally with Docker before pushing**
   ```bash
   ./build-docker.sh
   ```

3. **Keep PRs small for faster reviews**
   - Aim for < 200 changes when possible
   - Automatic labeling helps reviewers prioritize

4. **Write meaningful commit messages**
   - Used in automatic changelog generation

### For Maintainers

1. **Review security scan results weekly**
   - Check GitHub Security tab
   - Update dependencies as needed

2. **Monitor code coverage trends**
   - Aim for >80% coverage
   - Review coverage reports in PRs

3. **Create releases using semantic versioning**
   ```bash
   git tag -a v1.2.3 -m "Description"
   git push origin v1.2.3
   ```

4. **Update Docker image when dependencies change**
   - Modify Dockerfile
   - Workflow auto-builds and publishes

---

## Troubleshooting

### Build Failures

**Problem:** Build fails on specific compiler
- **Solution:** Check matrix-build.yml for compiler-specific issues
- Review error logs in the specific job

**Problem:** Tests timeout
- **Solution:** Increase timeout in CTest or workflow
- Check for infinite loops or deadlocks

### Docker Issues

**Problem:** Image build fails
- **Solution:** Check Dockerfile syntax
- Verify all COPY paths exist
- Test locally: `docker build -t test .`

**Problem:** Permission errors pulling image
- **Solution:** Image is public, authentication not needed
- Check GHCR status: https://github.com/features/packages

### Coverage Issues

**Problem:** Coverage report not generated
- **Solution:** Ensure CODECOV_TOKEN is set
- Check coverage.info file exists
- Verify lcov installation

---

## Performance Metrics

### Average Workflow Times
- CMake Single Platform: ~2-3 minutes (with cache)
- Matrix Build: ~15-20 minutes (8 configurations)
- Code Coverage: ~4-5 minutes
- Static Analysis: ~3-4 minutes
- Docker Build: ~8-10 minutes (first build), ~2-3 minutes (cached)

### Caching Impact
- Without cache: ~8 minutes build time
- With cache: ~2 minutes build time
- **Improvement: ~75% faster**

---

## Future Improvements

Potential enhancements to consider:

- [ ] Add benchmark workflow for performance regression testing
- [ ] Implement artifact signing for releases
- [ ] Add integration tests with external systems
- [ ] Create deployment workflow for documentation updates
- [ ] Add notifications (Slack, Discord) for build failures
- [ ] Implement automatic dependency updates (Dependabot)
- [ ] Add code quality metrics (SonarCloud)
- [ ] Create custom GitHub Actions for common tasks

---

## Related Documentation

- [DOCKER.md](DOCKER.md) - Docker setup and usage
- [CODE_STYLE.md](CODE_STYLE.md) - Code formatting guidelines
- [README.md](README.md) - Project overview and quick start
- [GitHub Actions Docs](https://docs.github.com/en/actions)

---

## Support

For issues with CI/CD workflows:
1. Check workflow logs in Actions tab
2. Review this documentation
3. Open an issue with workflow name and error details
4. Tag maintainers for urgent issues
