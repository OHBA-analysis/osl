#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import os
import inspect
import file_tree

class StudyTree:
    """Find and iterate through files in structured directories.

    https://pypi.org/project/file-tree/
    https://open.win.ox.ac.uk/pages/ndcn0236/file-tree/

    """
    def __init__(self, treename, study_dir):
        self.tree = file_tree.FileTree.read(treename)
        self.tree.top_level = study_dir

    def get(self, ftype, **kwargs):
        """Return list of files on disk that match type and conditions."""
        # Example use:
        # files = ft.get('fif', subject='sub001', task='restingstate')
        self.tree.update_glob(ftype, inplace=True)

        working_tree = self.tree.update(**kwargs)
        matches = [t.get(ftype) for t in working_tree.iter(ftype)]

        return matches
