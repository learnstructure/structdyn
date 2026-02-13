# Contributing to StructDyn

We love welcoming new contributors to StructDyn! Whether you're reporting a bug, suggesting a new feature, or writing code, your help is valued. This document outlines how to contribute.

## Code of Conduct

All participants in this project are expected to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the standards of behavior we expect.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue on our GitHub repository. A good bug report should include:

*   **A clear title:** "Bug: Newmark-Beta solver fails with zero damping."
*   **Your environment:** Python version, operating system.
*   **Steps to reproduce:** Provide a minimal code snippet that demonstrates the bug. This is crucial for us to be able to fix it.
*   **What you expected to happen:** Describe what the output should have been.
*   **What actually happened:** Include any error messages or incorrect results.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue to start a discussion. This allows us to coordinate efforts and ensure the suggestion fits with the project's goals.

### Your First Code Contribution

Unsure where to begin? Look for issues tagged with "good first issue" or "help wanted." These are tasks that we've identified as being good entry points for new contributors.

### Pull Request Process

1.  **Fork the repository:** Create your own copy of the project on GitHub.
2.  **Create a new branch:** `git checkout -b feature/my-new-feature` or `bugfix/fix-that-bug`.
3.  **Make your changes:** Write your code and add or update docstrings as needed.
4.  **Add tests:** If you've added new functionality, please add tests to the `tests/` directory to ensure it works correctly and prevent future regressions.
5.  **Run the tests:** Ensure that the entire test suite passes by running `pytest` (or your chosen test runner) from the root directory.
6.  **Format your code:** Ensure your code adheres to standard Python style guides (e.g., PEP 8).
7.  **Commit your changes:** Use a clear and descriptive commit message.
8.  **Push to your branch:** `git push origin feature/my-new-feature`.
9.  **Open a pull request:** Go to the main StructDyn repository and open a pull request from your forked repository's branch. Provide a clear description of the changes you have made.

Thank you for contributing to our project!
