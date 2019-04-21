#!/home/jalendesktop/anaconda3/envs/aseenv/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:04:51 2019
https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html
@author: jalencates
"""

from ase.build import bulk
from ase.calculators.espresso import Espresso
from ase.constraints import UnitCellFilter
from ase.optimize import LBFGS

pseudopotentials = {'Na': 'na_pbe_v1.5.uspp.F.UPF',
                    'Cl': 'cl_pbe_v1.4.uspp.F.UPF'}

rocksalt = bulk('NaCl', crystalstructure='rocksalt', a=6.0)
calc = Espresso(pseudopotentials=pseudopotentials,
                tstress=True, tprnfor=True, kpts=(3, 3, 3))

ucf = UnitCellFilter(rocksalt)
opt = LBFGS(ucf)
opt.run(fmax=0.005)

# cubic lattic constant
print((8*rocksalt.get_volume()/len(rocksalt))**(1.0/3.0))
