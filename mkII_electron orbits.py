import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import itertools
import time

def group_by_tolerance(evals, tol=0.01):
    sorted_idx = np.argsort(evals)
    groups = [[sorted_idx[0]]]
    for i in range(1, len(evals)):
        if evals[sorted_idx[i]] - evals[groups[-1][0]] < tol:
            groups[-1].append(sorted_idx[i])
        else:
            groups.append([sorted_idx[i]])
    return groups

def get_Oh_classes():
	"""Generates the 48 symmetry matrices of Oh and groups them by conjugacy class."""
	classes = {
		'E': [], '8C3': [], '3C2': [], '6C4': [], '6C2p': [],
		'i': [], '8S6': [], '3sigma_h': [], '6S4': [], '6sigma_d': []
	}
	for p in itertools.permutations([0, 1, 2]):
		for signs in itertools.product([1, -1], repeat=3):
			M = np.zeros((3, 3), dtype=int)
			for i in range(3):
				M[i, p[i]] = signs[i]

			det = int(np.round(np.linalg.det(M)))
			tr = int(np.round(np.trace(M)))
			nz_diag = np.sum(np.diag(M) != 0)

			if det == 1:
				if tr == 3:
					classes['E'].append(M)
				elif tr == 0:
					classes['8C3'].append(M)
				elif tr == 1:
					classes['6C4'].append(M)
				elif tr == -1:
					if nz_diag == 3:
						classes['3C2'].append(M)
					else:
						classes['6C2p'].append(M)
			else:
				if tr == -3:
					classes['i'].append(M)
				elif tr == 0:
					classes['8S6'].append(M)
				elif tr == -1:
					classes['6S4'].append(M)
				elif tr == 1:
					if nz_diag == 3:
						classes['3sigma_h'].append(M)
					else:
						classes['6sigma_d'].append(M)
	return classes


def get_Oh_character_table():
	char_table = {
		'A1g': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		'A2g': [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
		'Eg': [2, -1, 2, 0, 0, 2, -1, 2, 0, 0],
		'T1g': [3, 0, -1, 1, -1, 3, 0, -1, 1, -1],
		'T2g': [3, 0, -1, -1, 1, 3, 0, -1, -1, 1],
		'A1u': [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
		'A2u': [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
		'Eu': [2, -1, 2, 0, 0, -2, 1, -2, 0, 0],
		'T1u': [3, 0, -1, 1, -1, -3, 0, 1, -1, 1],
		'T2u': [3, 0, -1, -1, 1, -3, 0, 1, 1, -1]
	}
	class_order = ['E', '8C3', '3C2', '6C4', '6C2p', 'i', '8S6', '3sigma_h', '6S4', '6sigma_d']
	class_sizes = [1, 8, 3, 6, 6, 1, 8, 3, 6, 6]
	return char_table, class_order, class_sizes


def project_symmetry(evals, evecs, r2_array, L):
	"""Applies Oh character projection to rigorously classify eigenstates."""
	print("\n" + "=" * 80)
	print(" RIGOROUS O_h CHARACTER PROJECTION & PARITY CHECK")
	print("=" * 80)

	classes = get_Oh_classes()
	char_table, class_order, class_sizes = get_Oh_character_table()

	# Precompute permutation maps for all 48 matrices
	center = L // 2
	N = L ** 3 * 3

	x = np.arange(L) - center
	X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

	matrix_maps = {}
	for cls in class_order:
		matrix_maps[cls] = []
		for M in classes[cls]:
			M_inv = M.T
			c_old_map = [np.argmax(np.abs(M_inv[:, c])) for c in range(3)]

			X_old = M_inv[0, 0] * X + M_inv[0, 1] * Y + M_inv[0, 2] * Z + center
			Y_old = M_inv[1, 0] * X + M_inv[1, 1] * Y + M_inv[1, 2] * Z + center
			Z_old = M_inv[2, 0] * X + M_inv[2, 1] * Y + M_inv[2, 2] * Z + center

			idx_map = np.empty(N, dtype=int)
			for c in range(3):
				c_old = c_old_map[c]
				idx = c + 3 * (Z + center + L * (Y + center + L * (X + center)))
				idx_old = c_old + 3 * (Z_old + L * (Y_old + L * X_old))
				idx_map[idx.ravel()] = idx_old.ravel()

			matrix_maps[cls].append(idx_map)

	# Group eigenvalues by degeneracy
	groups = group_by_tolerance(evals, tol=0.01)

	print(f"{'Energy':>8} | {'Deg':>3} | ...")
	print("-" * 80)

	for group in groups:
		count = len(group)
		val = np.mean(evals[group])  # representative energy
		V = evecs[:, group]  # fancy-index the eigenvectors

		# Calculate radius
		avg_r2 = np.sum([np.sum((V[:, i] ** 2) * r2_array) for i in range(count)]) / count

		# Calculate traces
		chi = []
		for cls in class_order:
			class_tr = 0
			for idx_map in matrix_maps[cls]:
				tr = 0
				for i in range(count):
					tr += np.sum(V[:, i] * V[idx_map, i])
				class_tr += tr
			chi.append(class_tr / len(matrix_maps[cls]))

		# Project onto the standard character table
		proj = {}
		for irrep, chars in char_table.items():
			overlap = sum(class_sizes[i] * chars[i] * chi[i] for i in range(10)) / 48.0
			proj[irrep] = overlap

		components = []
		for irrep, overlap in proj.items():
			if overlap > 0.5:
				mult = int(np.round(overlap))
				if mult == 1:
					components.append(irrep)
				else:
					components.append(f"{mult}{irrep}")

		final_irrep = " \u2295 ".join(components) if components else "Artifact"

		# Translate irreducible representation back to atomic subshells
		if components == ['A1g']:
			subshell = 's (l=0)'
		elif components == ['T1u']:
			subshell = 'p (l=1) or f (l=3)'
		elif set(components) == {'Eg', 'T2g'}:
			subshell = 'd (l=2) (Full)'
		elif components == ['Eg'] or components == ['T2g']:
			subshell = 'd (l=2) component'
		elif set(components) == {'A2u', 'T1u', 'T2u'}:
			subshell = 'f (l=3) (Full)'
		elif components in [['A2u'], ['T2u']]:
			subshell = 'f (l=3) component'
		else:
			subshell = 'Mixed / Artifact'

		print(f"{val:8.3f} | {count:3d} | {avg_r2:7.2f} | {final_irrep:>22} | {subshell}")


def main():
	L = 35
	alpha = 10.0
	t_hop = 1.0
	t_mix = 20.0
	k = 30

	print(f"Building Walk Operator on {L}x{L}x{L} grid... (t_mix={t_mix})")
	t0 = time.time()

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

					for c_other in range(3):
						if c != c_other:
							rows.append(idx);
							cols.append(get_idx(x, y, z, c_other));
							data.append(-t_mix)

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
	print(f"Matrix constructed in {time.time() - t0:.2f}s.")

	print("\nDiagonalizing bound lepton states (sparse Lanczos)...")
	evals, evecs = eigsh(H, k=k, which='SA', tol=1e-6)

	project_symmetry(evals, evecs, r2_array, L)


if __name__ == "__main__":
	main()

'''
The Lanczos call eigsh(..., which='SA') returns eigenvalues already sorted ascending, so the helper doesn't need an argsort step. If you ever switch to which='SM' or call without sort, add sorted_idx = np.argsort(evals) at the top of the helper.
If tol=0.01 is too loose and starts merging genuinely distinct shells (you can see this if a "group" suddenly contains > 7 components, which is bigger than any single irrep), drop it to tol=0.005. Conversely if you still see split triplets, raise to tol=0.02. The right value is one that makes the deg counts line up to 1/2/3 cleanly across the spectrum.
'''