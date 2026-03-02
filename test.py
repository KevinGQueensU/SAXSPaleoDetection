from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

s = Structure.from_file("forsterite_symmetry.cif")  # 6-atom CIF with symmetry
# s already contains all sites after symmetry expansion; to force P1:
s_p1 = s.copy()
s_p1.modify_lattice(s.lattice)  # no change; just keep positions
CifWriter(s_p1, symprec=0, write_magmoms=False).write_file("forsterite_P1_28atoms.cif")