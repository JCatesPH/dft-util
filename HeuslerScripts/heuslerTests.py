#!/home/jalendesktop/anaconda3/envs/aseenv/bin python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:15:28 2019

@author: jalendesktop
"""

##
from ase import Atoms
from ase.spacegroup import crystal
import heuslerutil as heuslers

## Testing Full Heusler Utilities
# Make Full Heusler cell with desired X2YZ L21 structure
zr2fesifull = heuslers.makeFull16('Zr', 'Fe', 'Si', 6.545)
# zr2fesifull.write('zr2fesi16atom', 'vasp')

# Get the index of each atom, their positions, and the lattice param to check the object.
ions = zr2fesifull.get_chemical_symbols()
cellleng = zr2fesifull.get_cell()
cartpos = zr2fesifull.get_positions()
print("The Full Heusler Structure: Zr2FeSi")
print("===========================================================")
print("\n The ions are indexed in this order:")
print(ions)
print("\n The lattice parameters:")
print(cellleng)
print("\n The cartesian positions of the atoms:")
print(cartpos)
print("===========================================================")


# Make the disordered lattices and print them, then write POSCARs
#disorderSeriesMaker(zr2fesifull)


## Testing Inverse Heusler Utilities
# Make Inverse Heusler cell with desired X2YZ XA structure
zr2fesiIH = heuslers.makeInverse16('Zr', 'Fe', 'Si', 6.545)
# Get the index of each atom, their positions, and the lattice param to check the object.
ions = zr2fesiIH.get_chemical_symbols()
cellleng = zr2fesiIH.get_cell()
cartpos = zr2fesiIH.get_positions()
print("The Inverse Heusler Structure: Zr2FeSi")
print("===========================================================")
print("\n The ions are indexed in this order:")
print(ions)
print("\n The lattice parameters:")
print(cellleng)
print("\n The cartesian positions of the atoms:")
print(cartpos)
print("===========================================================")


## Testing Supercells
superZr2FeSiFull = heuslers.superFullH(zr2fesifull)
print("The Super Full Heusler Structure: Zr2FeSi")
print("===========================================================")
print("\n The ions are indexed in this order:")
print(ions)
print("\n The lattice parameters:")
print(cellleng)
print("\n The cartesian positions of the atoms:")
print(cartpos)
print("===========================================================")
#ions = zr2fesifull.get_chemical_symbols()
#cellleng = zr2fesifull.get_cell()
#cartpos = zr2fesifull.get_positions()

#superZr2FeSiFull.write('zr2fesi35atom', 'vasp')

print("\n \n The end of the script has been reached.")
print("===========================================================")
