# Contributing to StructDyn

First off, thank you for considering contributing to StructDyn! Your help is greatly appreciated. This document provides guidelines and instructions for contributing.

## Code of Conduct

All participants in this project are expected to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## Setting Up Your Development Environment

To ensure a consistent development experience, we recommend using a virtual environment.

1.  **Fork and clone the repository:**
    ```bash
    git clone https://github.com/learnstructure/structdyn.git
    cd structdyn
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS and Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Install the project in editable mode (`-e`) together with the test dependencies, and `flake8` for style checks.
    ```bash
    pip install -e ".[test]"
    pip install flake8
    ```

## How Can I Contribute?

There are many ways to contribute, from writing code to reporting bugs.

### Reporting Bugs

If you encounter a bug, please open an issue on GitHub. A helpful bug report includes:

- **A clear and descriptive title** – e.g., “Bug: Newmark‑Beta solver fails with zero damping”
- **Your environment** – Python version, operating system, and relevant package versions (`numpy`, `scipy`, etc.)
- **A minimal, reproducible code example**
- **The expected behaviour** and **what actually happened** (including error messages or incorrect results)

### Suggesting Enhancements

Have an idea for a new feature or an improvement? Open an issue to discuss it first. This helps avoid duplication of effort and ensures the suggestion aligns with the project’s scope.

### Your First Code Contribution

If you're looking for a place to start, check out issues labelled **“good first issue”** or **“help wanted”**.

## Pull Request Process

1.  **Set up your environment** as described above.
2.  **Create a new branch** for your work: `git checkout -b feature/my-new-feature` or `bugfix/fix-that-bug`.
3.  **Make your changes.** Write your code and add or update docstrings as needed.
4.  **Add tests.** If you've added new functionality, please add tests to the `tests/` directory to ensure it works correctly and prevent future regressions.
5.  **Check code style and run tests.** Before committing, ensure your code is well-formatted and that all tests pass.
    ```bash
    # Check code style with flake8
    flake8 .

    # Run the test suite with pytest
    pytest
    ```
6.  **Record the change.** Add a line under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
7.  **Commit your changes.** Use a clear and descriptive commit message.
8.  **Push to your branch:** `git push origin feature/my-new-feature`.
9.  **Open a pull request.** Go to the main StructDyn repository and open a pull request from your forked repository's branch. Provide a clear description of the changes you have made, referencing any relevant issues. The test suite runs automatically on Linux, Windows, and macOS for every pull request, and it must pass before the pull request is merged.

## Getting Help

- **Questions about using `structdyn`**: ask in [GitHub Discussions](https://github.com/learnstructure/structdyn/discussions), or open an issue if you think the documentation is unclear or wrong.
- **Bugs and feature requests**: open an [issue](https://github.com/learnstructure/structdyn/issues) as described above.
- **Documentation**: [structdyn.readthedocs.io](https://structdyn.readthedocs.io/en/latest/).

The maintainer aims to respond to new issues and pull requests within two weeks.

## Governance

`structdyn` is maintained by Abinash Mandal ([@learnstructure](https://github.com/learnstructure)), who reviews and merges pull requests and publishes releases.

- **Scope.** `structdyn` analyses the dynamic response of single- and multi-degree-of-freedom systems with established methods of structural dynamics, implemented so that they can be checked against textbook examples. Finite element model building belongs to companion packages such as `fem2d` and `fem3d`, which pass their stiffness and mass matrices to `structdyn`.
- **Decisions.** Changes to the public API or to the scope are proposed and discussed in a public issue before they are implemented. Breaking changes are marked as such in the changelog.
- **Correctness.** Every new analysis method needs a test that reproduces a published example or an independent solution.
- **Releases.** Versions follow [semantic versioning](https://semver.org/). Each release is recorded in [CHANGELOG.md](CHANGELOG.md), tagged on GitHub, published on PyPI, and archived on Zenodo.
- **Maintainers.** Contributors with sustained contributions may be invited to become maintainers.

## Licensing of Contributions

By contributing to StructDyn, you agree that your contributions will be licensed under its [MIT License](LICENSE).

---

Thank you for contributing to our project!
