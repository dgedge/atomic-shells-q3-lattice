"""
Test the Stage 1 Madelung verification on the Z^3 ⊗ Q_3 lattice.

Verifies that the cubic substrate's geometric E_g compression generates a
sufficient core-penetration penalty (first-order Hartree on a Neon-like
10-electron core) to flip the bare 4s > 3d_eg ordering into Madelung's
4s < 3d_eg ordering.

This is the directional check on the geometric origin of Madelung's rule.
A full self-consistent Hartree-Fock with exchange is left to a follow-up;
Stage 1 confirms only that the proposed mechanism (E_g compression →
core overlap → penalty differential → ordering flip) works in the
expected direction.

Note: first-order Hartree without exchange overshoots for compact orbitals
because of self-interaction. This test therefore verifies *only* the
4s/3d_eg crossover, not the full (n+ℓ) Madelung sequence. The 3p < 4s
piece of Madelung requires self-interaction correction or full HF and
is not asserted here.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _build_bare_hamiltonian(L, alpha, t_hop, t_mix):
    """Construct the bare Z^3 ⊗ Q_3 tight-binding Hamiltonian.

    Returns (H_csr, r2_array, coords, site_to_states) where:
      - H_csr is the (3L^3, 3L^3) sparse Hamiltonian
      - r2_array[s] = squared lattice distance from origin for state s
      - coords[i] = (dx, dy, dz) of site i (in lattice units, centred at origin)
      - site_to_states[i, c] = global state index for site i, sublattice c
    """
    center = L // 2
    num_sites = L ** 3
    num_states = num_sites * 3

    def get_idx(x, y, z, c):
        return c + 3 * (z + L * (y + L * x))

    rows, cols, data = [], [], []
    r2_array = np.zeros(num_states)
    coords = np.zeros((num_sites, 3))
    site_to_states = np.zeros((num_sites, 3), dtype=int)

    for x in range(L):
        for y in range(L):
            for z in range(L):
                dx, dy, dz = x - center, y - center, z - center
                r2 = dx ** 2 + dy ** 2 + dz ** 2
                r = np.sqrt(r2)
                V = -alpha * 2.0 if r == 0 else -alpha / r

                site_idx = z + L * (y + L * x)
                coords[site_idx] = [dx, dy, dz]

                for c in range(3):
                    state_idx = get_idx(x, y, z, c)
                    site_to_states[site_idx, c] = state_idx
                    r2_array[state_idx] = r2
                    rows.append(state_idx); cols.append(state_idx); data.append(V)

                    for c_other in range(3):
                        if c != c_other:
                            rows.append(state_idx)
                            cols.append(get_idx(x, y, z, c_other))
                            data.append(-t_mix)

                    if c == 0:
                        if x > 0:
                            rows.append(state_idx); cols.append(get_idx(x - 1, y, z, c)); data.append(-t_hop)
                        if x < L - 1:
                            rows.append(state_idx); cols.append(get_idx(x + 1, y, z, c)); data.append(-t_hop)
                    elif c == 1:
                        if y > 0:
                            rows.append(state_idx); cols.append(get_idx(x, y - 1, z, c)); data.append(-t_hop)
                        if y < L - 1:
                            rows.append(state_idx); cols.append(get_idx(x, y + 1, z, c)); data.append(-t_hop)
                    elif c == 2:
                        if z > 0:
                            rows.append(state_idx); cols.append(get_idx(x, y, z - 1, c)); data.append(-t_hop)
                        if z < L - 1:
                            rows.append(state_idx); cols.append(get_idx(x, y, z + 1, c)); data.append(-t_hop)

    H = sp.csr_matrix((data, (rows, cols)), shape=(num_states, num_states))
    return H, r2_array, coords, site_to_states


def _compute_neon_core_density(evecs, site_to_states, n_filled_spatial=5):
    """Build the 10-electron Neon-like core density from the lowest n_filled_spatial
    spatial states, doubled by spin capacity.

    For the bipyramid lattice this is 1s + 2s + 2p (1 + 1 + 3 = 5 spatial states,
    occupied with 2 electrons each → 10-electron core).
    """
    num_sites = site_to_states.shape[0]
    density = np.zeros(num_sites)
    for i in range(n_filled_spatial):
        vec = evecs[:, i]
        for c in range(3):
            density += 2.0 * (vec[site_to_states[:, c]] ** 2)
    return density


def _compute_hartree_potential(density, coords, alpha, batch_size=500):
    """Compute V_HF(r) = α ∫ ρ(r') / |r - r'| d³r' on each lattice site.

    The on-site self-term uses 2.0 (a regularised stand-in for 1/0).
    Batched O(N²) evaluation; for L=31 this is ~30k × 30k → manageable."""
    num_sites = density.shape[0]
    V_HF = np.zeros(num_sites)
    for i in range(0, num_sites, batch_size):
        end = min(i + batch_size, num_sites)
        diff = coords[i:end, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist == 0] = 1.0
        inv_dist = 1.0 / dist
        np.fill_diagonal(inv_dist, 2.0)
        V_HF[i:end] = np.sum(inv_dist * density[np.newaxis, :] * alpha, axis=1)
    return V_HF


def _state_avg_r2(vec, r2_array):
    return float(np.sum((vec ** 2) * r2_array))


def _state_penalty(vec, V_HF, site_to_states, num_states):
    """First-order penalty <ψ| V_HF |ψ> for a single eigenstate."""
    vhf_state = np.zeros(num_states)
    for c in range(3):
        vhf_state[site_to_states[:, c]] = V_HF
    return float(np.sum((vec ** 2) * vhf_state))


def _group_by_tolerance(evals, tol=0.01):
    """Group sorted eigenvalues into degenerate multiplets within tol."""
    groups = [[0]]
    for i in range(1, len(evals)):
        if evals[i] - evals[groups[-1][0]] < tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


@pytest.mark.slow
def test_madelung_4s_3d_eg_crossover():
    """Stage 1: cubic E_g compression flips 4s below 3d_eg under first-order Hartree.

    The mechanism: 3d_eg is geometrically compressed by the cubic substrate to
    <r^2> ≈ 1.0 (Shell-2 territory), while 4s sits diffusely at <r^2> ≈ 5.
    The 10-electron Neon-like core then exerts a much larger Coulomb penalty
    on the compact 3d_eg than on the diffuse 4s, flipping the bare ordering.
    """
    L = 31
    alpha, t_hop, t_mix = 10.0, 1.0, 20.0
    k = 30

    # 1. Bare spectrum
    H, r2_array, coords, site_to_states = _build_bare_hamiltonian(L, alpha, t_hop, t_mix)
    num_states = H.shape[0]
    evals, evecs = eigsh(H, k=k, which='SA', tol=1e-5)

    # 2. Neon-like core density from 5 lowest spatial states
    density = _compute_neon_core_density(evecs, site_to_states, n_filled_spatial=5)

    # 3. First-order Hartree potential
    V_HF = _compute_hartree_potential(density, coords, alpha)

    # 4. Group eigenvalues into multiplets
    sorted_idx = np.argsort(evals)
    sorted_evals = evals[sorted_idx]
    sorted_evecs = evecs[:, sorted_idx]
    groups = _group_by_tolerance(sorted_evals, tol=0.01)

    # 5. Find 3d_eg (lowest E_g doublet at compact <r^2>) and 4s (lowest diffuse singlet)
    target_3d_eg = None
    target_4s = None

    for group in groups:
        deg = len(group)
        E_bare = float(np.mean(sorted_evals[group]))
        avg_r2 = float(np.mean([_state_avg_r2(sorted_evecs[:, i], r2_array) for i in group]))
        avg_pen = float(np.mean([_state_penalty(sorted_evecs[:, i], V_HF, site_to_states, num_states)
                                 for i in group]))

        # 3d_eg: doublet, pulled inward by cubic compression to <r^2> < 1.5
        if deg == 2 and avg_r2 < 1.5 and target_3d_eg is None:
            target_3d_eg = {
                "E_bare": E_bare, "r2": avg_r2,
                "V_HF": avg_pen, "E_dressed": E_bare + avg_pen,
            }

        # 4s: singlet at large <r^2> (diffuse, well outside the core)
        if deg == 1 and avg_r2 > 4.0 and target_4s is None:
            target_4s = {
                "E_bare": E_bare, "r2": avg_r2,
                "V_HF": avg_pen, "E_dressed": E_bare + avg_pen,
            }

    assert target_3d_eg is not None, (
        "Could not locate 3d_eg state (expected E_g doublet at <r^2> < 1.5). "
        "Spectrum may be missing the cubic compression signature."
    )
    assert target_4s is not None, (
        "Could not locate 4s state (expected A_1g singlet at <r^2> > 4). "
        "Box may be too small to bind 4s."
    )

    # === ASSERTIONS ===

    # (i) Bare ordering: 4s sits ABOVE 3d_eg in bare spectrum (standard hydrogenic order)
    assert target_4s["E_bare"] > target_3d_eg["E_bare"], (
        f"Bare ordering wrong: 4s (E={target_4s['E_bare']:.3f}) should sit "
        f"above 3d_eg (E={target_3d_eg['E_bare']:.3f}) in the bare lattice "
        f"spectrum. The Madelung crossover claim depends on this baseline."
    )

    # (ii) Penalty differential: compressed 3d_eg suffers larger core penalty than diffuse 4s
    assert target_3d_eg["V_HF"] > target_4s["V_HF"], (
        f"Core-penetration penalty wrong: compressed 3d_eg (V_HF="
        f"{target_3d_eg['V_HF']:.3f}) should suffer a larger penalty than "
        f"diffuse 4s (V_HF={target_4s['V_HF']:.3f}). The geometric mechanism "
        f"requires the E_g compression to drive the differential."
    )

    # (iii) Madelung crossover: dressed 4s drops below dressed 3d_eg
    assert target_4s["E_dressed"] < target_3d_eg["E_dressed"], (
        f"Madelung crossover failed: dressed 4s (E="
        f"{target_4s['E_dressed']:.3f}) should sit below dressed 3d_eg "
        f"(E={target_3d_eg['E_dressed']:.3f}) after first-order Hartree. "
        f"The proposed geometric origin of Madelung's rule does not survive."
    )

    # (iv) Margin check: the flip is substantial (not marginal)
    margin = target_3d_eg["E_dressed"] - target_4s["E_dressed"]
    assert margin > 5.0, (
        f"Madelung margin {margin:.2f} too small (< 5 lattice units). "
        f"The crossover is marginal, suggesting the mechanism is not robust "
        f"to small parameter changes."
    )


if __name__ == "__main__":
    # Allow direct execution for development inspection (not just pytest).
    test_madelung_4s_3d_eg_crossover()
    print("Madelung Stage 1 test passed: 4s drops below 3d_eg under first-order Hartree.")