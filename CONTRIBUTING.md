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

### Installation

To install an editable version with pip:
```
conda create --name osl python=3
git clone git@github.com:OHBA-analysis/oslpy.git
cd oslpy
pip install -e .
```

### Tests

To run all tests:
```
cd osl
pytest tests
```
or to run a specific test, e.g. file handling:
```
cd osl/tests
pytest test_file_handling.py
```

### Build Documentation

To build the documentation:
```
python setup.py build_sphinx
```
Compiled docs can be found in `doc/build/html/index.html`.

### Bug Fixes & New features

To fix an isolated problem, any contributor can.

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

# git school

Practical overview of workflows in git - I'm assuming you're roughly familiar with adding and committing before starting this.

First clone the repo into your chosen directory

```
git clone https://github.com/OHBA-analysis/oslpy.git
cd oslpy
```

Now you're ready to start making changes.

## Creating & switching branches

New work must be created on a 'branch' - a separate copy of the code which allows you to make changes away from the main code.

#### Create a new branch based on the current main:
This will create, but not switch to, a new branch
```
git branch mynewbranch
```
#### Switch to your branch:
This will switch to an existing branch
```
git checkout mynewbranch
```
####  create a new branch and switch to it in one step:
This will create and switch to a new branch.
```
git checkout -b  mynewbranch
```

### Checking and updating your local repo

Once you have a branch you can start making changes and adding/committing files to your local copy.

#### Check the current state of your local branch
At any point, you can run this for a quick summary of your branches state
```
git status
```
This will display your local branch, whether it is up-to-date with any remote branches, any staged/unstaged changes and any untracked files all in one place.

If someone might have changed the branch whilst you've been working (or even just to be sure), you can check for any upstream changes with:
```
git remote update
```

This will check the server for changes **but won't merge any changes into your local repo** so it is safe to run with work in progress. If you run `git status` again after a remote update - you may see that the info has changed. In particular, you may now be behind the remote branch. If you are behind the remote branch you can add those changes into your repo with:
```
git pull
```
This may create a conflict if files have changed both locally and on the remote... This can be fixed locally and finalised with a 'merge commit'

### Pushing changes back to the remote

If theree are no remaining conflicts, `git remote  update` has been run and `git status` shows that you are **ahead** of the remote - you can push your local changes  with:

```
git push origin mynewbrach
```


### Branch management

If you're working on an idea or trying out something new, then the work can stay on a branch for as long as you need it. It is recommended to keep your branch up-to-date with main where possible. You can merge the current main branch into your branch with:
```
git remote update
git merge origin main
```

This may create conflicts... these should be fixed before carrying on with other work. Once any conflicts are fixed,  your local branch is now up to date with main. You can push these changes back to your remote branch with:

```
git push origin mynewbranch
```

### Pull requests

Once you're ready to share you're changes, or would like a place to track changes and tests, you  can create a pull request. You can do this on the website by nagvigating to the branch page at

```
https://github.com/OHBA-analysis/oslpy/tree/mynewbranch
```
There should be a grey bar below the branch name saying 'This branch is 5 commits ahead of main' or similar. If you click the "Contribute" button in the right of this box - you can create a pull request.
