Contributing to OSL-Python
==========================

OSL accepts bug reports (as issues), bug fixes (as pull requests) and new features (as pull requests).


## Issues (Bug Reports)
If you have found a bug or problem with the code, please [open an issue](https://github.com/OHBA-analysis/oslpy/issues/new) on github. An EXCELLENT bug report contains the following ingredients:

* **Summary** A concise description of the the bug.
* **Versions and context** Which OSL/Python versions (eg OSL v0.0.1 on Python 3.9) and which operating system (eg Debian 11.2, Mac OSX 10.15 or Windows 10) are you using?
* **Steps to reproduce** How can we reproduce the issue? - this is very important. Please use (`) to format inline code and (```) to format code blocks
* **What is the expected correct behavior?** What should happen instead of this current behaviour?
* **Relevant logs and/or screenshots** Paste any relevant logs - please use code blocks (```) to format console output, logs, and code.
* **Possible fixes** If you can, link to the line of code that might be responsible for the problem, any suggestions for fixes are welcome!

Small and simple bugs might not need all this info but generally a more thorough description will help find a quick solution.

## Pull Requests (Bug Fixes and New Features)

If you want to contribute new code or documentation to OSL - you'll need to open a 'pull request'. There are two ways to do this.

### Pull request from fork

If you have developed a new features or bug fix on your own copy (fork) of OSL and would like to contribute it back to OSL. Then you can [create a pull request from your forked version of osl](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

Please create the pull request with the **base** branch set to **main** on oslpy and the **compare** branch set to your contribution. Please include an informative title and a clear description of what changes you have made and what the expected behaviour should be. An OSL-Python developer will review the code and potentially ask for some changes before the code can be merged into OSL. In some cases a pull-request may be rejected.

### Pull request from OSL branch

If you are an OSL developer, or working with the OSL team, you can make contributions by creating a new branch directly on the OSL repository. Please create the branch relative to **main** and ensure that it stays up to date with subsequent changes to the **main** branch. Once you're ready to start sharing the branch. You can make a [pull request directly from the branch on OSL](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). If you would like to discuss and share changes but not merge them yet - please create a 'draft pull request'.

## Developer Notes

The core developer team is @woolrich, @ajquinn, @cgohil8 and @matsvanes. A few example workflows are listed below.

### Bug Fixes & New features

To fix an isolated problem, any contributer can.

* create a fix on a branch of oslpy:main
* create a pull request with a description and links to relevant issues.
* get feedback/approval from a core developer
* make any required changes

Once changes are made and all tests pass, a core developer can complete the merge.

### Hotfixes

For urgent and obvious fixes, a core developer can do the following:

* create a fix on a branch of oslpy:main
* create a pull request with quick description of the issue
* merge it to main themselves.
