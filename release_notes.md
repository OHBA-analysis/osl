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

Run tests on your local machine, from your root osl directory with:

```
conda activate osl
pytest osl/tests
```

Passes and warnings are fine. Just need to check for failures.

Note, you may need to install `pytest` with `pip install pytest` to run the tests.


##### 3 - Increment the version numbers in across the codebase

You need to update the version number the the following files to version number you'd like to release:

- `setup.py`
- `osl/__init__.py`

Don't forget to commit these changes before continuing.

##### 4 - Run a local build and make sure you get the right versions.

Run a local install using pip from the local directory:

```
pip install .
```

The final line of output should say 'Successfully installed' and include osl with the correct version number, if its incorrect then check the version in `setup.py`

Next, start a python session, import osl and check `osl.__version__` - this should show the correct version number, it it is incorrect then check the version in `osl/__init__.py`


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

Note, you may need to install `twine` with `pip install twine`.

##### 8 - TEST EVERYTHING!

Ask your friends and family to install the released package and let you know if there are any problems. Fix any that come up and repeat steps 1 to 8 with a new version number. 
Note: if you test the installation from a `pip install osl`, make sure you're not opening Python from an OSL directory because then the directory will be imported rather than the installed package from pip.

Do not delete broken package releases, make any fixes in a new 'trivial' update.


##### 9 - Update version numbers to include 'dev'

The same versions you incremented in step 3 should be updated to increment once more and include 'dev' at the end, this means we will be able to distinguish the tagged/fixed version from future work-in-progress. If we don't do this, then updates to main will have the same version even though they are likely to significantly differ from the release.

If we just did a minor update and released v0.3.0, we would make the development version v0.4.dev0.

If we did a trivial update and released v0.1.2, we would still fix the development version to the next minor release v0.2.dev0. Don't bother incrementing the trivial versions.

You need to change the version number in the following files:

- `setup.py`
- `osl/__init__.py`

Don't forget to commit these changes before continuing.

##### 10 - Push branch and merge into main

To push use:

```
git push --set-upstream origin release-vX.Y.Z
```
Remember to replace vX.Y.Z with the latest version number.

Then create the pull request.

Wait for someone to point out which mistake you made. Fix it and then repeat steps 1 to 10 with an appropriate new version number.

Once you're happy and if the tests pass you can merge.
