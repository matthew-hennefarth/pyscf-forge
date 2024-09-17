from pyscf.lib import logger
import numpy as np
from pyscf.data import nist
from pyscf import lib
from pyscf.prop.dip_moment import lpdft
from pyscf.grad import lpdft as lpdft_grad
from pyscf.prop.dip_moment.mcpdft import mcpdft_HellmanFeynman_dipole, get_guage_origin

#Not sure if we even need this line below?
#from pyscf.grad.mspdft import mspdft_heff_response

class TransitionDipole (lpdft.ElectricDipole):

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        val = np.linalg.norm(mol_dip)
        i   = self.state[0]
        j   = self.state[1]
        dif = abs(self.e_states[i]-self.e_states[j])
        osc = 2/3*dif*val**2
        if unit.upper() == 'DEBYE':
            for x in [ham_response, LdotJnuc, mol_dip]: x *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('L-PDFT TDM <{}|mu|{}>          {:>10} {:>10} {:>10}'.format(i,j,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Transition Dipole Moment (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        log.note('Oscillator strength  : %9.5f', osc)
        return mol_dip

    # lpdft dip_moment get_ham_response from pyscf-forge/pyscf/prop/dip_moment/lpdft.py
    # removed the nuclear term
    def get_ham_response(self, state=None, verbose=None, mo=None,
            ci=None, origin='Coord_Center', **kwargs):
        if state is None: state   = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo      = self.base.mo_coeff
        if ci is None: ci      = self.base.ci

        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]

        elec_term = mcpdft_HellmanFeynman_dipole (fcasscf, mo_coeff=mo, ci=ci[state], origin=origin)
#       nucl_term = nuclear_dipole(fcasscf, origin=origin)
#       total = nucl_term + elec_term
        return elect_term
