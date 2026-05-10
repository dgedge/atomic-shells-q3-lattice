import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# L was 25
def test_color_singlet_shells(L=35, alpha=10.0, t_hop=1.0, t_mix=20.0):
	"""
	Evaluates the W operator on Z³ ⊗ Q³ with a strong color-mixing
	term (t_mix) to enforce the Color-Singlet (Lepton) constraint,
	pushing colored doublets out of the low-energy spectrum.
	Computes <r^2> to verify spatial shells.
	"""
	center = L // 2
	num_sites = L ** 3
	num_states = num_sites * 3

	def get_idx(x, y, z, c):
		return c + 3 * (z + L * (y + L * x))

	rows, cols, data = [], [], []
	r2_array = np.zeros(num_states)

	for x in range(L):
		for y in range(L):
			for z in range(L):
				dx, dy, dz = x - center, y - center, z - center
				r2 = dx ** 2 + dy ** 2 + dz ** 2
				r = np.sqrt(r2)
				V = -alpha * 2.0 if r == 0 else -alpha / r

				for c in range(3):
					idx = get_idx(x, y, z, c)
					r2_array[idx] = r2
					rows.append(idx);
					cols.append(idx);
					data.append(V)

					# Q3 Mixing. Large t_mix acts as confinement penalty,
					# enforcing the symmetric color-singlet ground state.
					for c_other in range(3):
						if c != c_other:
							rows.append(idx);
							cols.append(get_idx(x, y, z, c_other));
							data.append(-t_mix)

					# Inter-cell directional hopping
					if c == 0:
						if x > 0: rows.append(idx); cols.append(get_idx(x - 1, y, z, c)); data.append(-t_hop)
						if x < L - 1: rows.append(idx); cols.append(get_idx(x + 1, y, z, c)); data.append(-t_hop)
					elif c == 1:
						if y > 0: rows.append(idx); cols.append(get_idx(x, y - 1, z, c)); data.append(-t_hop)
						if y < L - 1: rows.append(idx); cols.append(get_idx(x, y + 1, z, c)); data.append(-t_hop)
					elif c == 2:
						if z > 0: rows.append(idx); cols.append(get_idx(x, y, z - 1, c)); data.append(-t_hop)
						if z < L - 1: rows.append(idx); cols.append(get_idx(x, y, z + 1, c)); data.append(-t_hop)

	H = sp.csr_matrix((data, (rows, cols)), shape=(num_states, num_states))

	print("\nDiagonalizing bound lepton states (sparse Lanczos)...")
	evals, evecs = eigsh(H, k=50, which='SA', tol=1e-6)

	unique_evals, counts = np.unique(np.round(evals, 3), return_counts=True)

	print("\n--- COLOR-SINGLET BOUND STATES (Leptons) ---")
	print(f"{'Energy':>8} | {'Deg':>3} | {'<r^2>':>7} | {'O_h Irrep (Deduced Shell)'}")
	print("-" * 55)

	idx_eval = 0
	for val, count in zip(unique_evals, counts):
		avg_r2 = 0
		for _ in range(count):
			vec = evecs[:, idx_eval]
			avg_r2 += np.sum((vec ** 2) * r2_array)
			idx_eval += 1
		avg_r2 /= count

		irrep = "A1/A2 (Singlet)" if count == 1 else "E (Doublet)" if count == 2 else "T1/T2 (Triplet)" if count == 3 else "Artifact"

		print(f"{val:8.3f} | {count:3d} | {avg_r2:7.2f} | {irrep}")


if __name__ == "__main__":
	test_color_singlet_shells()