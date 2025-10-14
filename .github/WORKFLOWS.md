# GitHub Actions Workflows

This repository uses GitHub Actions for continuous integration and deployment. Below is an overview of all workflows.

## 📊 Workflow Status Dashboard

| Workflow | Status | Frequency | Purpose |
|----------|--------|-----------|---------|
| [CMake Single Platform](.github/workflows/cmake-single-platform.yml) | [![CMake Build](../../actions/workflows/cmake-single-platform.yml/badge.svg)](../../actions/workflows/cmake-single-platform.yml) | On Push/PR | Primary build and test |
| [Matrix Build](.github/workflows/matrix-build.yml) | [![Matrix Build](../../actions/workflows/matrix-build.yml/badge.svg)](../../actions/workflows/matrix-build.yml) | On Push/PR | Multi-compiler testing |
| [Code Coverage](.github/workflows/code-coverage.yml) | [![Code Coverage](../../actions/workflows/code-coverage.yml/badge.svg)](../../actions/workflows/code-coverage.yml) | On Push/PR | Coverage analysis |
| [Static Analysis](.github/workflows/static-analysis.yml) | [![Static Analysis](../../actions/workflows/static-analysis.yml/badge.svg)](../../actions/workflows/static-analysis.yml) | On Push/PR | Code quality checks |
| [Format Check](.github/workflows/format-check.yml) | [![Code Formatting](../../actions/workflows/format-check.yml/badge.svg)](../../actions/workflows/format-check.yml) | On Push/PR | Code style verification |
| [Docker Publish](.github/workflows/docker-publish.yml) | [![Docker](../../actions/workflows/docker-publish.yml/badge.svg)](../../actions/workflows/docker-publish.yml) | On Docker changes | Image building |
| [Dependency Scan](.github/workflows/dependency-scan.yml) | [![Security Scan](../../actions/workflows/dependency-scan.yml/badge.svg)](../../actions/workflows/dependency-scan.yml) | Weekly + Push/PR | Security scanning |
| [Documentation](.github/workflows/documentation.yml) | [![Documentation](../../actions/workflows/documentation.yml/badge.svg)](../../actions/workflows/documentation.yml) | On Push/PR | Docs generation |
| [PR Labels](.github/workflows/pr-labels.yml) | [![PR Labeling](../../actions/workflows/pr-labels.yml/badge.svg)](../../actions/workflows/pr-labels.yml) | On PR | Automated labeling |
| [Release](.github/workflows/release.yml) | [![Release](../../actions/workflows/release.yml/badge.svg)](../../actions/workflows/release.yml) | On Tag | Release automation |

## 🔍 Quick Actions

### View All Workflows
[Go to Actions Tab](../../actions)

### Manually Trigger Workflows
Some workflows can be manually triggered:
- [Docker Build](../../actions/workflows/docker-publish.yml)
- [Documentation Generation](../../actions/workflows/documentation.yml)
- [Release Creation](../../actions/workflows/release.yml)

### View Recent Runs
- [Recent Workflow Runs](../../actions/runs)
- [Failed Runs](../../actions/runs?status=failure)

## 📖 Documentation

For detailed information about each workflow, see [CI_CD.md](../CI_CD.md).

## 🛠️ Troubleshooting

If a workflow fails:
1. Click on the failing workflow badge above
2. Review the error logs
3. Check [CI_CD.md](../CI_CD.md#troubleshooting) for common issues
4. Open an issue if you need help

## 🔒 Secrets Management

Required secrets (configured in repository settings):
- `GITHUB_TOKEN` - Automatically provided
- `CODECOV_TOKEN` - For code coverage uploads (optional)

## 📊 Metrics

Current workflow performance:
- Average build time: ~2-3 minutes (with cache)
- Matrix build time: ~15-20 minutes
- Cache hit rate: ~80%
- Build speed improvement: ~75% with caching
