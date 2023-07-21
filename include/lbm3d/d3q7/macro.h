#pragma once

template < typename TRAITS >
struct D3Q7_MACRO_Default
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum { e_phi, N };

	static const bool use_syncMacro = false;

	// called from LBMKernelInit
	template < typename LBM_KS >
	CUDA_HOSTDEV static void zeroForcesInKS(LBM_KS &KS)
	{
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		SD.macro(e_phi, x, y, z) = KS.phi;
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
	}
};


template < typename TRAITS >
struct D3Q7_MACRO_Void
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	static const int N = 0;

	// called from LBMKernelInit
	template < typename LBM_KS >
	CUDA_HOSTDEV static void zeroForcesInKS(LBM_KS &KS)
	{
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
	}
};
