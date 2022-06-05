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


import re
import glob
import parse
from string import Formatter

class Study:

    def __init__(self, studydir):
        self.studydir = studydir

        # Extract field names in between {braces}
        self.fieldnames = [fname for _, fname, _, _ in Formatter().parse(self.studydir) if fname]

        # Replace braces with wildcards
        self.globdir = re.sub("\{.*?\}","*", studydir)

        self.match_files = sorted(glob.glob(self.globdir))
        print('found {} files'.format(len(self.match_files)))

        self.match_values = []
        for fname in self.match_files:
            self.match_values.append(parse.parse(self.studydir, fname).named)

        self.fields = {}
        # Use first file as a reference for keywords
        for key, value in self.match_values[0].items():
            self.fields[key] = [value]
            for d in self.match_values[1:]:
                self.fields[key].append(d[key])

    def get(self, check_exist=True, **kwargs):
        keywords = {}
        for key in self.fieldnames:
            keywords[key] = kwargs.get(key, '*')

        fname = self.studydir.format(**keywords)

        return sorted(glob.glob(fname))
