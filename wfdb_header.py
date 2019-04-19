#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:15:47 2018

@author: chenhx1992
"""

import glob, os
import shutil
import re

os.chdir('./training_raw/')

for file in glob.glob('*.hea'):
    with open(file) as from_file:

        line = from_file.readline().strip()

        # make any changes to line here
        p = re.compile('(?P<name>\S+)\s(?P<num>\S+)\s(?P<freq>\S+)\s(?P<length>\S+)\s(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)\s(?P<hr>\d+):(?P<min>\d+):(?P<sec>\d+)')
        info = p.search(line)

        new_line = '{} {} {} {} {}:{}:{} {}/{}/{} \n'.format(info.group('name'), info.group('num'), info.group('freq'), info.group('length'), info.group('hr'), info.group('min'), info.group('sec'), info.group('day'), info.group('month'), info.group('year'))

        with open(file, 'w') as to_file:
            to_file.write(new_line)
            shutil.copyfileobj(from_file, to_file)