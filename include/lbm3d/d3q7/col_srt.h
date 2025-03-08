#include "common.h"

template <typename TRAITS, typename LBM_EQ = D3Q7_EQ<TRAITS>>
struct D3Q7_SRT : D3Q7_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "SRT";

	template <typename LBM_KS>
	CUDA_HOSTDEV static void collision(LBM_KS& KS)
	{
		const dreal tau = n1o2 + LBM_EQ::iCs2 * KS.lbmViscosity;
		const dreal omega = no1 / tau;
		KS.f[zzz] += omega * (LBM_EQ::eq_zzz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[zzz]);
		KS.f[pzz] += omega * (LBM_EQ::eq_pzz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[pzz]);
		KS.f[zpz] += omega * (LBM_EQ::eq_zpz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[zpz]);
		KS.f[zzp] += omega * (LBM_EQ::eq_zzp(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[zzp]);
		KS.f[mzz] += omega * (LBM_EQ::eq_mzz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[mzz]);
		KS.f[zmz] += omega * (LBM_EQ::eq_zmz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[zmz]);
		KS.f[zzm] += omega * (LBM_EQ::eq_zzm(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[zzm]);
	}
};
