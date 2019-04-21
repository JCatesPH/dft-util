#!/home/jalendesktop/anaconda3/envs/aseenv/bin/python
# -*- coding: utf-8 -*-
"""
# Created on Thu Apr  4 20:15:28 2019

@author: jalendesktop
"""

# Environment setting
from ase import Atoms
import numpy as np
from ase.visualize import view

def makeFull16( X, Y, Z, cellParam ):
    """
    Takes input for heusler lattice to create 16 atom cell. Allows manipulation to make slabs, defects, etc.

    Inputs:
    X : Atomic symbol of X site atom
    Y : Atomic symbol of Y site atom
    Z : Atomic symbol of Z site atom
    cellparam : cell parameter
    Outputs:
    L21cell : ase.Atoms object with 16 atoms
    """
    a = cellParam
    L21cell = Atoms((X, X, X, X, X, X, X, X, Y, Y, Y, Y, Z, Z, Z, Z),
                positions = [(0,0,0), (.5*a,0,0), (0,.5*a,0), (.5*a,.5*a,0), (0,0,.5*a), (.5*a,0,.5*a), (0,.5*a,.5*a), (.5*a,.5*a,.5*a),
                             (.25*a,.25*a,.25*a), (.75*a, .75*a, .25*a), (.75*a, .25*a, .75*a), (.25*a,.75*a,.75*a),
                             (.75*a,.25*a,.25*a), (.25*a,.75*a,.25*a), (.25*a,.25*a,.75*a), (.75*a,.75*a,.75*a)],
                cell = (cellParam, cellParam, cellParam))
    return L21cell

def superFullH(acell16):
    """
    Takes the 16 atom cell as input, then performs the translations to get supercell with the same X, Y, Z, and cell param.
    """
    ions = acell16.get_chemical_symbols()
    X = ions[0]
    a = acell16.get_cell()[0,0]
    w = a/2
    AddedX = Atoms((X, X, X,  X, X, X,  X, X, X,  X, X, X,  X, X, X,  X, X, X, X), 
                   positions = [(0,a,0), (a,0,0), (a,w,0),
                                (a,a,0), (w,a,0), (0,a,w),
                                (a,0,w), (a,w,w), (a,a,w),
                                (w,a,w), (0,0,a), (0,w,a),
                                (0,a,a), (w,0,a), (a,0,a),
                                (w,w,a), (a,w,a), (a,a,a),
                                (w,a,a)])
    return acell16.extend(AddedX)

def swapSites(acell16, disorder):
    """
    The function takes the L21 cell and desired disorder, then it swaps the sites in the cell to get the desired POSCAR file.
    
    Input:
        acell16 : L21 cell as ase.Atoms object
        disorder : string with values '12.5', '25', '37.5', or '50' describing disorder percentage
    Output:
        disorderedcell : A new ase.Atoms object with desired disorder
    """
    copyofacell16 = acell16
    if (disorder=='12.5'):
        copyofacell16.positions[[8,4]] = copyofacell16.positions[[4,8]]
    elif (disorder=='25'):
        copyofacell16.positions[[8,4]] = copyofacell16.positions[[4,8]]
        copyofacell16.positions[[8,4]] = copyofacell16.positions[[4,8]]
    elif (disorder=='37.5'):
        copyofacell16.positions[[8,4]] = copyofacell16.positions[[4,8]]
    elif (disorder=='50'):
        copyofacell16.positions[[8,4]] = copyofacell16.positions[[4,8]]        
    else : print('Please enter acceptable disorder amount: "12.5", "25", "37.5", or "50"')
    
    return copyofacell16

##

zr2fesifull = makeFull16('Zr', 'Fe', 'Si', 6.545)
zr2fesifull.write('zr2fesi16atom', 'vasp')

ions = zr2fesifull.get_chemical_symbols()
cellleng = zr2fesifull.get_cell()
cartpos = zr2fesifull.get_positions()



zr2fesifull.positions[[0,8]] = zr2fesifull.positions[[8,0]]
zr2fesifull.write('zr2fesi16atom08defect', 'vasp')
cartpos1 = zr2fesifull.get_positions()

zr2fesifull.positions[[8,0]] = zr2fesifull.positions[[0,8]]
zr2fesifull.positions[[8,1]] = zr2fesifull.positions[[1,8]]
zr2fesifull.write('zr2fesi16atom18defect', 'vasp')
cartpos2 = zr2fesifull.get_positions()

zr2fesifull.positions[[8,1]] = zr2fesifull.positions[[1,8]]
zr2fesifull.positions[[8,2]] = zr2fesifull.positions[[2,8]]
zr2fesifull.write('zr2fesi16atom28defect', 'vasp')
cartpos3 = zr2fesifull.get_positions()

zr2fesifull.positions[[8,2]] = zr2fesifull.positions[[2,8]]
zr2fesifull.positions[[8,3]] = zr2fesifull.positions[[3,8]]
zr2fesifull.write('zr2fesi16atom38defect', 'vasp')
cartpos4 = zr2fesifull.get_positions()

zr2fesifull.positions[[8,3]] = zr2fesifull.positions[[3,8]]
zr2fesifull.positions[[8,4]] = zr2fesifull.positions[[4,8]]
zr2fesifull.write('zr2fesi16atom48defect', 'vasp')
cartpos5 = zr2fesifull.get_positions()

zr2fesifull.positions[[9,7]] = zr2fesifull.positions[[7,9]]
zr2fesifull.write('zr2fesi16atom4879defect', 'vasp')
cartpos6 = zr2fesifull.get_positions()

#superZr2FeSiFull = superFullH(zr2fesifull)

#ions = zr2fesifull.get_chemical_symbols()
#cellleng = zr2fesifull.get_cell()
#cartpos = zr2fesifull.get_positions()

#superZr2FeSiFull.write('zr2fesi35atom', 'vasp')