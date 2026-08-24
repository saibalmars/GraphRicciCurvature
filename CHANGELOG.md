# Changelog

All notable changes to GraphRicciCurvature are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Trusted PyPI publishing from GitHub Releases using OpenID Connect.
- GitHub Pages documentation deployment.

### Changed

- Pairwise shortest-path queries are batched by source to avoid repeated
  Dijkstra searches while retaining the mode's low-memory, non-APSP behavior.
- Package metadata and dependencies are defined in `pyproject.toml`.
- Package versions are derived from Git tags, so publishing `v0.5.4` produces
  package version `0.5.4` without editing source files.

### Fixed

- Weighted Ollivier-Ricci normalization now uses the graph shortest-path
  metric.

## [0.5.3.2] - 2024-06-26

See the [GitHub release notes](https://github.com/saibalmars/GraphRicciCurvature/releases/tag/v0.5.3.2).

[Unreleased]: https://github.com/saibalmars/GraphRicciCurvature/compare/v0.5.3.2...HEAD
[0.5.3.2]: https://github.com/saibalmars/GraphRicciCurvature/releases/tag/v0.5.3.2
