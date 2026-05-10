import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.signal import fftconvolve


def run_stage1_madelung():
	L = 31
	center = L // 2
	num_sites = L ** 3
	num_states = num_sites * 3
	alpha, t_hop, t_mix = 10.0, 1.0, 20.0

	def get_idx(x, y, z, c):
		return c + 3 * (z + L * (y + L * x))

	rows, cols, data = [], [], []
	r2_array = np.zeros(num_states)
	coords = np.zeros((num_sites, 3))
	site_to_states = np.zeros((num_sites, 3), dtype=int)

	# 1. Build the bare Z^3 x Q3 Hamiltonian
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
					rows.append(state_idx);
					cols.append(state_idx);
					data.append(V)

					for c_other in range(3):
						if c != c_other:
							rows.append(state_idx);
							cols.append(get_idx(x, y, z, c_other));
							data.append(-t_mix)

					if c == 0:
						if x > 0: rows.append(state_idx); cols.append(get_idx(x - 1, y, z, c)); data.append(-t_hop)
						if x < L - 1: rows.append(state_idx); cols.append(get_idx(x + 1, y, z, c)); data.append(-t_hop)
					elif c == 1:
						if y > 0: rows.append(state_idx); cols.append(get_idx(x, y - 1, z, c)); data.append(-t_hop)
						if y < L - 1: rows.append(state_idx); cols.append(get_idx(x, y + 1, z, c)); data.append(-t_hop)
					elif c == 2:
						if z > 0: rows.append(state_idx); cols.append(get_idx(x, y, z - 1, c)); data.append(-t_hop)
						if z < L - 1: rows.append(state_idx); cols.append(get_idx(x, y, z + 1, c)); data.append(-t_hop)

	H_bare = sp.csr_matrix((data, (rows, cols)), shape=(num_states, num_states))
	print("Diagonalizing Bare Hamiltonian...")
	evals, evecs = eigsh(H_bare, k=30, which='SA', tol=1e-5)

	# 2. Extract 10-electron Core Density (Lowest 5 spatial states: 1s, 2s, 2p)
	print("Calculating Neon Core Density...")
	density = np.zeros(num_sites)
	for i in range(5):
		vec = evecs[:, i]
		for c in range(3):
			density += 2.0 * (vec[site_to_states[:, c]] ** 2)  # x2 for Spin capacity

	# 3. Calculate 1/r Hartree Repulsion manually
	print("Calculating V_HF Repulsion Matrix...")
	V_HF = np.zeros(num_sites)
	batch_size = 500
	for i in range(0, num_sites, batch_size):
		end = min(i + batch_size, num_sites)
		diff = coords[i:end, np.newaxis, :] - coords[np.newaxis, :, :]
		dist = np.linalg.norm(diff, axis=2)
		dist[dist == 0] = 1.0
		inv_dist = 1.0 / dist
		np.fill_diagonal(inv_dist, 2.0)
		V_HF[i:end] = np.sum(inv_dist * density[np.newaxis, :] * alpha, axis=1)

	# 4. Evaluate First-Order Overlap Penalty <psi | V_HF | psi>
	print("\n--- STAGE 1: FIRST ORDER CORE PENALTY ---")
	unique_evals, counts = np.unique(np.round(evals, 2), return_counts=True)
	idx = 0

	print(f"{'Subshell':>10} | {'E_bare':>8} | {'<r^2>':>6} | {'Core Penalty <V_HF>':>20}")
	print("-" * 54)

	for val, count in zip(unique_evals, counts):
		avg_r2, avg_vhf = 0, 0
		for _ in range(count):
			vec = evecs[:, idx]
			avg_r2 += np.sum((vec ** 2) * r2_array)
			vhf_state = np.zeros(num_states)
			for c in range(3):
				vhf_state[site_to_states[:, c]] = V_HF
			avg_vhf += np.sum((vec ** 2) * vhf_state)
			idx += 1

		avg_r2 /= count
		avg_vhf /= count

		if count == 1 and avg_r2 < 0.1:
			label = "1s"
		elif count == 1 and 1.0 < avg_r2 < 1.2:
			label = "2s"
		elif count == 3 and 1.0 < avg_r2 < 1.2:
			label = "2p"
		elif count == 2 and 1.0 < avg_r2 < 1.2:
			label = "3d (Eg)"
		elif count == 1 and avg_r2 > 4.0:
			label = "4s"
		else:
			label = None

		if label:
			print(f"{label:>10} | {val:8.2f} | {avg_r2:6.2f} | {avg_vhf:20.2f}")


if __name__ == "__main__":
	run_stage1_madelung()