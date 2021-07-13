#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import inspect
from fsl.utils.filetree import FileTree, tree_directories, FileTreeQuery


class StudyTree:

    def __init__(self, treename, study_dir):
        tree_directories.append(inspect.getfile(inspect.currentframe()))
        self.tree = FileTree.read(treename, study_dir)
        self.query = FileTreeQuery(self.tree)

    def get(self, template, asmatch=False):
        if template not in self.query.templates:
            raise ValueError

        qq = self.query.query(template)
        if asmatch is False:
            qq = [q.filename for q in qq]

        return qq
