# Authors: Andrew Quinn <a.quinn@bham.ac.uk>

import re
import glob
import parse
from string import Formatter

class Study:
    """Class for simple file finding and looping."""

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
