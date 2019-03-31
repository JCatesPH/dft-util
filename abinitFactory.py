#!/home/jalendesktop/anaconda3/envs/abienv/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:26:08 2019
https://nbviewer.jupyter.org/github/abinit/abitutorials/blob/master/abitutorials/input_factories.ipynb
@author: jalencates
"""
# %%

from __future__ import division, print_function, unicode_literals

import os
import sys

import abipy.abilab as abilab
import abipy.data as abidata
import abipy.flowtk as flowtk
from pprint import pprint
import numpy as np

# %%
cifFile = "/home/jalendesktop/Documents/DFT/dftfiles/StructureFiles/Cr2FeSi_IH_nounit.cif"
#pseudo_Fe = "/home/jalendesktop/Documents/DFT/dftfiles/Pseudos/Fe.psp8"
#pseudo_Cr = "/home/jalendesktop/Documents/DFT/dftfiles/Pseudos/Cr.psp8"
#pseudo_Si = "/home/jalendesktop/Documents/DFT/dftfiles/Pseudos/Si.psp8"
pseudo_dir = "/home/jalendesktop/Documents/DFT/dftfiles/pseudos/"
print(pseudo_dir)
cr2fesi=abilab.Structure.from_file(cifFile)
print(cr2fesi)

inp = abilab.AbinitInput(structure=cr2fesi, pseudos={14,24,26}, pseudo_dir=psuedo_dir)
