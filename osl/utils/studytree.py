#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import os
import inspect
from fsl.utils.filetree import FileTree, tree_directories, FileTreeQuery


class StudyTree:

    def __init__(self, treename, study_dir):
        tree_directories.append(os.path.dirname(inspect.getfile(inspect.currentframe())))
        self.study_dir = study_dir
        self.tree = FileTree.read(treename, study_dir)
        self.query = FileTreeQuery(self.tree)

    def get(self, template, asmatch=False, strip_fname=False, **kwargs):
        if template not in self.query.templates:
            raise ValueError(
                "template not found. Available: {0}\n".format(" ".join(self.query.templates))
                + "Using base directory: {0}".format(self.study_dir)
            )

        qq = self.query.query(template, **kwargs)
        if asmatch is False:
            qq = [q.filename for q in qq]

        if strip_fname is True:
            qq = [os.path.dirname(q) for q in qq]

        return qq
