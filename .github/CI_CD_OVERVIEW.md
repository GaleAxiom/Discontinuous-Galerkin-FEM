# CI/CD Pipeline Overview

Quick visual reference for the DG-FEM CI/CD pipeline.

## 📊 Workflow Execution Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         GitHub Event                              │
│  (Push to main, Pull Request, Tag, Schedule, Manual)             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼─────┐   ┌────▼─────┐   ┌────▼─────┐
    │  Build   │   │   Test   │   │ Quality  │
    │ Workflows│   │Workflows │   │ Checks   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
    ┌────▼─────────────────────────────▼──────┐
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  CMake Single Platform           │   │
    │  │  • Build with Docker             │   │
    │  │  • Run tests                     │   │
    │  │  • Cache builds (75% faster)     │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Matrix Build                    │   │
    │  │  • GCC 11, 12                    │   │
    │  │  • Clang 14, 15                  │   │
    │  │  • Ubuntu 20.04, 22.04           │   │
    │  │  • Debug + Release               │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Code Coverage                   │   │
    │  │  • Generate lcov reports         │   │
    │  │  • Upload to Codecov             │   │
    │  │  • HTML artifacts                │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Static Analysis                 │   │
    │  │  • cppcheck (bugs)               │   │
    │  │  • clang-tidy (quality)          │   │
    │  │  • Upload reports                │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Format Check                    │   │
    │  │  • Verify clang-format           │   │
    │  │  • Fail if incorrect             │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    └──────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼─────┐   ┌────▼─────┐   ┌────▼─────┐
    │ Security │   │  Deploy  │   │  Utils   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
    ┌────▼──────────────▼──────────────▼──────┐
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Dependency Scan                 │   │
    │  │  • Trivy vulnerability scan      │   │
    │  │  • SARIF upload                  │   │
    │  │  • Weekly schedule               │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Docker Publish                  │   │
    │  │  • Build multi-stage images      │   │
    │  │  • Push to GHCR                  │   │
    │  │  • Tag: latest, dev, prod        │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Documentation                   │   │
    │  │  • Generate Doxygen docs         │   │
    │  │  • Deploy to GitHub Pages        │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  Release (on tag)                │   │
    │  │  • Generate changelog            │   │
    │  │  • Create GitHub release         │   │
    │  │  • Package artifacts             │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    │  ┌──────────────────────────────────┐   │
    │  │  PR Labels                       │   │
    │  │  • Auto-size labels              │   │
    │  │  • Auto-type labels              │   │
    │  └──────────────────────────────────┘   │
    │                                          │
    └──────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Success │
                    │    ✓    │
                    └─────────┘
```

## 🔄 Workflow Triggers

| Workflow | Push | PR | Tag | Schedule | Manual |
|----------|------|-----|-----|----------|--------|
| CMake Single Platform | ✓ | ✓ | - | - | - |
| Matrix Build | ✓ | ✓ | - | - | - |
| Code Coverage | ✓ | ✓ | - | - | - |
| Static Analysis | ✓ | ✓ | - | - | - |
| Format Check | ✓ | ✓ | - | - | - |
| Docker Publish | ✓* | - | - | - | ✓ |
| Dependency Scan | ✓ | ✓ | - | Weekly | - |
| Documentation | ✓ | ✓ | - | - | ✓ |
| Release | - | - | ✓ | - | ✓ |
| PR Labels | - | ✓ | - | - | - |

*Only when Dockerfile or workflow changes

## 📈 Performance Metrics

```
Build Time Comparison:

Without Cache:  ████████████████████  ~8 min
With Cache:     █████  ~2 min (75% faster!)

Docker Image Sizes:

Development:    ████████████████████████████████████████████  850 MB
Production:     █████████████  250 MB (70% smaller!)
```

## 🎯 Quick Commands

### Local Development
```bash
# Start dev environment
docker-compose up dev

# Run tests
docker-compose up test

# Build release
docker-compose up build
```

### Manual Workflow Triggers
```bash
# Via GitHub UI:
# Actions → [Workflow Name] → Run workflow

# Or via gh CLI:
gh workflow run docker-publish.yml
gh workflow run documentation.yml
```

### View Workflow Status
```bash
# List recent runs
gh run list --limit 10

# View specific run
gh run view [run-id]

# Download artifacts
gh run download [run-id]
```

## 📚 Documentation Quick Links

- **Complete CI/CD Docs:** [CI_CD.md](../CI_CD.md)
- **Docker Guide:** [DOCKER.md](../DOCKER.md)
- **Contribution Guide:** [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Workflow Status:** [WORKFLOWS.md](WORKFLOWS.md)
- **Improvements Summary:** [IMPROVEMENTS_SUMMARY.md](../IMPROVEMENTS_SUMMARY.md)

## 🔍 Monitoring Checklist

**Daily:**
- [ ] Check failing workflows
- [ ] Review PR labels

**Weekly:**
- [ ] Review security scan results
- [ ] Check code coverage trends
- [ ] Review static analysis reports

**Monthly:**
- [ ] Update dependencies
- [ ] Review and clean old artifacts
- [ ] Optimize workflow performance

## 🚨 Troubleshooting

**Build fails on PR:**
1. Check format: `./format-code.sh`
2. Build locally: `./build-docker.sh`
3. Review error logs in Actions tab

**Docker image won't pull:**
1. Image is public, no auth needed
2. Check network: `docker pull ubuntu:22.04`
3. Try different tag: `:dev`, `:prod`, `:main-[sha]`

**Coverage not uploading:**
1. Check `CODECOV_TOKEN` secret is set
2. Verify coverage.info exists
3. Review Codecov dashboard

## 📊 Success Criteria

✅ **All workflows passing**  
✅ **Coverage > 80%**  
✅ **No high-severity security alerts**  
✅ **Build time < 3 minutes (with cache)**  
✅ **All PRs auto-labeled**  
✅ **Documentation up to date**

---

Last Updated: 2025-10-13  
For detailed information, see [CI_CD.md](../CI_CD.md)
