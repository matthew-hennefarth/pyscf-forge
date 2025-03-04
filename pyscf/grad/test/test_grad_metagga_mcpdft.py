#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>

import unittest

from pyscf import scf, gto, df, dft, mcscf
from pyscf.data.nist import BOHR
from pyscf import mcpdft

from mrh.my_pyscf.fci import csf_solver

def diatomic(
    atom1,
    atom2,
    r,
    fnal,
    basis,
    ncas,
    nelecas,
    nstates,
    charge=None,
    spin=None,
    symmetry=False,
    cas_irrep=None,
    density_fit=False,
    grids_level=9
):
    """Used for checking diatomic systems to see if the Lagrange Multipliers are working properly."""
    global mols
    xyz = "{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0".format(atom1, atom2, r)
    mol = gto.M(
        atom=xyz,
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=symmetry,
        verbose=0,
        output="/dev/null",
    )
    mols.append(mol)
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df.aug_etb(mol))

    mc = mcpdft.CASSCF(mf.run(), fnal, ncas, nelecas, grids_level=grids_level)
    if spin is None:
        spin = mol.nelectron % 2

    # ss = spin * (spin + 2) * 0.25
    # mc.fix_spin_(ss=ss, shift=2)
    mc.fcisolver = csf_solver(mol, smult=spin+1)

    if nstates > 1:
        mc = mc.state_average(
            [
                1.0 / float(nstates),
            ]
            * nstates,
        )

    mc.conv_tol = 1e-12
    mc.conv_grad_tol = 1e-6
    mo = None
    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep(cas_irrep)

    mc_grad = mc.run(mo).nuc_grad_method()
    mc_grad.conv_rtol = 1e-12
    return mc_grad


def setUpModule():
    global mols, original_grids
    mols = []
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False


def tearDownModule():
    global mols, diatomic, original_grids
    [m.stdout.close() for m in mols]
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    del mols, diatomic, original_grids


class KnownValues(unittest.TestCase):

    def test_grad_lih_sstm06l22_sto3g(self):
        mc = diatomic("Li", "H", 0.8, "tM06L", "STO-3G", 2, 2, 1, grids_level=1)
        de = mc.kernel()[1, 0] / BOHR

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   4015363355dc691a80bc94d4b2b094318b213e36
        DE_REF = -1.0546009263404388

        self.assertAlmostEqual(de, DE_REF, 5)

    def test_grad_lih_sa2tm06l22_sto3g(self):
        mc = diatomic("Li", "H", 0.8, "tM06L", "STO-3G", 2, 2, 2, grids_level=1)

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   4015363355dc691a80bc94d4b2b094318b213e36
        DE_REF = [-1.0351271000, -0.8919881992]

        for state in range(2):
            with self.subTest(state=state):
                de = mc.kernel(state=state)[1, 0] / BOHR
                self.assertAlmostEqual(de, DE_REF[state], 5)

    def test_grad_lih_ssmc2322_sto3g(self):
        mc = diatomic("Li", "H", 0.8, "MC23", "STO-3G", 2, 2, 1, grids_level=1)
        de = mc.kernel()[1, 0] / BOHR

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   5027df09b644c819439046f9bb34b7efee2ac3e0
        DE_REF = -1.0641645070

        self.assertAlmostEqual(de, DE_REF, 5)


    def test_custom(self):
        df = False
        r = 0.8
        mc = diatomic("Li", "H", r, "MC23", "STO-3G", 2, 2, 1, grids_level=1, density_fit=df)

        de_ana = mc.kernel(state=0)[1,0]/BOHR

        mc_scanner = mc.base.as_scanner()
        print(de_ana)

        import numpy as np

        deltas = []
        de_num = []

        for i in np.arange(2, 8, 0.1):
            delta = np.exp(-i)
            deltas.append(delta)
            mc_scanner(f"Li 0 0 0; H {r+delta} 0 0")
            e1 = mc_scanner.e_tot
            mc_scanner(f"Li 0 0 0; H {r-delta} 0 0")
            e2 = mc_scanner.e_tot
            de = (e1-e2)/(2*delta) 

            de_num.append(de)
            # print(de-de_ana)

        print(",".join([f"{d:.10f}" for d in deltas]))
        print(",".join([f"{d:.10f}" for d in de_num]))


if __name__ == "__main__":
    print("Full Tests for MC-PDFT gradients with meta-GGA functionals")
    unittest.main()
