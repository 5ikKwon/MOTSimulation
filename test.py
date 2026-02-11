import numpy as np
import pylcp
import pylcp.fields as fields
import scipy.constants as cts
import matplotlib.pyplot as plt

atom = pylcp.atom("87Rb")
# Set unitless mass
mass = (atom.state[2].gamma*atom.mass)/(cts.hbar*(100*2*np.pi*atom.transition[1].k)**2)

det = -2.0
s = 1.0
H_g_D2, mu_q_g_D2 = pylcp.hamiltonians.hyperfine_coupled(
    atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
    atom.state[0].Ahfs/atom.state[2].gammaHz, Bhfs=0, Chfs=0,
    muB=1)

H_e_D2, mu_q_e_D2 = pylcp.hamiltonians.hyperfine_coupled(
    atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
    Ahfs=atom.state[2].Ahfs/atom.state[2].gammaHz,
    Bhfs=atom.state[2].Bhfs/atom.state[2].gammaHz, Chfs=0,
    muB=1)

dijq_D2 = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(
    atom.state[0].J, atom.state[2].J, atom.I)

E_e_D2 = np.unique(np.diagonal(H_e_D2))
E_g_D2 = np.unique(np.diagonal(H_g_D2))
det_cooling = E_e_D2[3]-E_g_D2[1]+det # F:2->F':3
det_repumping = E_e_D2[2]-E_g_D2[0] # F:1->F':2
hamiltonian_D2 = pylcp.hamiltonian(H_g_D2, H_e_D2, mu_q_g_D2, mu_q_e_D2, dijq_D2, mass = mass)
hamiltonian_D2.print_structure()
print(E_e_D2)
print(E_g_D2)
# mF level indexing is basically assending order
hamiltonian_D2.return_full_H(np.array([0., 0., 0.]), np.array([0., 0., 0.]))

# Change the laserBeam polarization. For outgoing magnetic field, we need pairs of CW wave
laserBeams_cooling_D2 = pylcp.laserBeams([
        {'kvec':np.array([1., 0., 0.]), 'pol':+1, 'delta':det_cooling, 's':s, 'wa':1e3, 'wb':2e3},
        {'kvec':np.array([-1., 0., 0.]), 'pol':+1, 'delta':det_cooling, 's':s, 'wa':1e3, 'wb':2e3},
        {'kvec':np.array([0., 1., 0.]), 'pol':-1, 'delta':det_cooling, 's':s, 'wa':1e3, 'wb':2e3},
        {'kvec':np.array([0., -1., 0.]), 'pol':-1, 'delta':det_cooling, 's':s, 'wa':1e3, 'wb':2e3}
        ], beam_type=pylcp.ellipticalgaussianBeam)
laserBeams_repumping_D2 = pylcp.laserBeams([
        {'kvec':np.array([1., 0., 0.]), 'pol':+1, 'delta':det_repumping, 's':s, 'wa':1e3, 'wb':2e3},
        {'kvec':np.array([-1., 0., 0.]), 'pol':+1, 'delta':det_repumping, 's':s, 'wa':1e3, 'wb':2e3},
        {'kvec':np.array([0., 1., 0.]), 'pol':-1, 'delta':det_repumping, 's':s, 'wa':1e3, 'wb':2e3},
        {'kvec':np.array([0., -1., 0.]), 'pol':-1, 'delta':det_repumping, 's':s, 'wa':1e3, 'wb':2e3}
        ], beam_type=pylcp.ellipticalgaussianBeam)
laserBeams_D2 = laserBeams_cooling_D2
alpha = 1
magfield = pylcp.iPMagneticField(B0=0, B1=alpha, B2=0)
obe = pylcp.obe(laserBeams_D2, magfield, hamiltonian_D2, include_mag_forces=True, transform_into_re_im=True)

r0 = np.array([0,0,0])
obe.set_initial_position(r0)
obe.set_initial_velocity(np.zeros((3,)))
obe.set_initial_rho_from_rateeq()

tmax = 1e2
teval = np.linspace(0, tmax, 5001)
# I had to change method and relax tolerance to make solver stable.
# It might not be needed when we put a realistic numbers
# obe.evolve_motion(
#     [0, tmax],
#     t_eval=teval,
#     atol=1e-6, rtol=1e-4,
#     random_recoil=False,
#     progress_bar=True,
#     max_scatter_probability=0.5,
#     record_force=True,
#     events = ()
# )

print("beta =", obe.damping_coeff(axes=[1]))
print("omega =", obe.trapping_frequencies())