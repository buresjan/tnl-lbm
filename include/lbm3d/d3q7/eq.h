#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

template <typename TRAITS>
struct D3Q7_EQ
{
	using dreal = typename TRAITS::dreal;

	// iCs2 = 1/Cs^2, where Cs is the lattice speed of sound
	static constexpr dreal iCs2 = 4;

	// weights depending on Cs - see Kruger page 85 eq (3.60)
	static constexpr dreal w_0 = n1o4;	// central
	static constexpr dreal w_1 = n1o8;	// non-central

	__cuda_callable__ static dreal feq(int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		// general second order Maxwell-Boltzmann Equilibrium
		return no1 - n1o2 * iCs2 * (vx * vx + vy * vy + vz * vz) + iCs2 * (qx * vx + qy * vy + qz * vz)
			 + n1o2 * iCs2 * iCs2 * (qx * vx + qy * vy + qz * vz) * (qx * vx + qy * vy + qz * vz);
	}

	__cuda_callable__ static dreal eq_zzz(dreal phi, dreal vx, dreal vy, dreal vz)
	{
		return w_0 * phi * feq(0, 0, 0, vx, vy, vz);
	}
	__cuda_callable__ static dreal eq_pzz(dreal phi, dreal vx, dreal vy, dreal vz)
	{
		return w_1 * phi * feq(1, 0, 0, vx, vy, vz);
	}
	__cuda_callable__ static dreal eq_zpz(dreal phi, dreal vx, dreal vy, dreal vz)
	{
		return w_1 * phi * feq(0, 1, 0, vx, vy, vz);
	}
	__cuda_callable__ static dreal eq_zzp(dreal phi, dreal vx, dreal vy, dreal vz)
	{
		return w_1 * phi * feq(0, 0, 1, vx, vy, vz);
	}
	__cuda_callable__ static dreal eq_mzz(dreal phi, dreal vx, dreal vy, dreal vz)
	{
		return w_1 * phi * feq(-1, 0, 0, vx, vy, vz);
	}
	__cuda_callable__ static dreal eq_zmz(dreal phi, dreal vx, dreal vy, dreal vz)
	{
		return w_1 * phi * feq(0, -1, 0, vx, vy, vz);
	}
	__cuda_callable__ static dreal eq_zzm(dreal phi, dreal vx, dreal vy, dreal vz)
	{
		return w_1 * phi * feq(0, 0, -1, vx, vy, vz);
	}
};
