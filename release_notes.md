OHBA Software Library (OSL) Release Notes
=========================================
Information about managing releases of OSL

General Workflow
----------------

We are going to prepare release vX.Y.Z - for a real release this should contain three numerical values indicating the new version number. eg v0.1.2 or v3.4.7.

The third digit should be incremented for trivial changes and bugfixes that don't affect the API
The second digit should be incremented for minor changes, could include some API changes
The first digit should be incremented for major changes and additions that shift things a lot and may not be backwards compatible

The lower digits reset to zero if a higher digit increments, for example a moderate update to v0.2.4 could become v0.3.0.

The codebase release versions may contain 'dev' in them - please ignore this for the release itself, this is to distinguish code installed from source with formal releases during debugging, you'll add this back in step 7.

Replace vX.Y.Z with the correct release version from here on out!!

##### 0 - Ensure relevant branches are merged to main

Have a look at the current work-in-progress, is there something that should be included in a new release?


##### 1 - Create a branch and pull request for the release

This can be done as normal using your favourite method. Give both a helpful name like 'release-vX.Y.Z etc

Make sure that you create the release branch from the current main branch - not a subbranch, this is easily done in a rush.

```
git checkout main
git remote update
git pull
git checkout -b release-vX.Y.Z
```

##### 2 - Make sure the tests pass

These should be run on github automatically as part of the pull request, you
can also run them with `pytest` in the osl root directory. Fix anything that
fails - though nothing should at this stage....


##### 3 - Increment the version numbers in across the codebase

Verson numbers are included in a number of files - these ALL need to be updated, though only one or two are absolutely critical. `setup.py` and `osl/__init__.py` are vital and must match or someone is going to get very confused later on. I'd strongly recommend running `git grep version` to search for other possibles.


##### 4 - Run a local build and make sure you get the right versions.

Run a local install using pip from the local directory:

```
pip install .
```

The final line of output should say 'Successfully installed' and include osl with the correct version number, if its incorrect then check the version in `setup.py`

Next, start a python session, import osl and check osl.__version__ - this should show the correct version number, it it is incorrect then check the version in `osl/__init__.py`


##### 5 - tag a new version

Use git to tag the current branch state with an informative message (that must match the correct version number....) and push it to github.

```
git tag -a vX.Y.Z -m "bump to vX.Y.Z"
git push origin vX.Y.Z
```

##### 6 - Create release on github

Follow instructions here to publish the release on github

https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository

##### 7 - Push release to PyPi.org

Create a build/wheel for the tagged version by running

```
python setup.py sdist
python setup.py bdist_wheel --universal
```

and upload to PyPi.org using twine, ensure you have your username and password to hand.

```
twine upload --skip-existing dist/*
```

##### 8 - TEST EVERYTHING!

Ask your friends and family to install the released pacakge and let you know if there are any problems. Fix any that come up and repeat steps 1 to 8 with a new version number.

Do not delete broken package releases, make any fixes in a new 'trivial' update.


##### 9 - Update version numbers to include 'dev'

The same versions you incremented in step 3 should be updated to increment once mor and include 'dev' at the end, this means we will be able to distinguish the tagged/fixed version from future work-in-progress. If we don't do this, then updates to main will have the same version even though they are likely to significantly differ from the release.

If we just did a minor update and released v0.3.0, we would make the development version v0.4.dev0.

##### 10 - Merge the branch into main

And wait for someone to point out which mistake you made. Fix it and then repeat steps 1 to 10 with an appropriate new version number.
