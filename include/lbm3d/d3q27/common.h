#pragma once

template <
	typename T_TRAITS,
	typename T_EQ
>
struct D3Q27_COMMON
{
	using TRAITS = T_TRAITS;
	using EQ = T_EQ;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;

	template< typename LBM_KS >
	CUDA_HOSTDEV static void computeDensityAndVelocity(LBM_KS &KS)
	{
		#ifdef USE_HIGH_PRECISION_RHO
		// src: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
		KS.rho = 0;
		dreal c = 0;                 // A running compensation for lost low-order bits.
		for (int i=0;i<27;i++)
		{
			dreal y = KS.f[i] - c;
			dreal t = KS.rho + y;
			c = (t-KS.rho)-y;
			KS.rho = t;
		}
		#else
		// based on Geier 2015 Appendix J
		KS.rho= ((((KS.f[ppp]+KS.f[mmm]) + (KS.f[pmp]+KS.f[mpm])) + ((KS.f[ppm]+KS.f[mmp])+(KS.f[mpp]+KS.f[pmm])))
			+(((KS.f[zpp]+KS.f[zmm]) + (KS.f[zpm]+KS.f[zmp])) + ((KS.f[pzp]+KS.f[mzm])+(KS.f[pzm]+KS.f[mzp])) + ((KS.f[ppz]+KS.f[mmz]) + (KS.f[pmz]+KS.f[mpz])))
			+((KS.f[pzz]+KS.f[mzz]) + (KS.f[zpz]+KS.f[zmz]) + (KS.f[zzp]+KS.f[zzm]))) + KS.f[zzz];
		#endif

		computeVelocity(KS);
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void computeVelocity(LBM_KS &KS)
	{
		KS.vz=((((KS.f[ppp]-KS.f[mmm])+(KS.f[mpp]-KS.f[pmm]))+((KS.f[pmp]-KS.f[mpm])+(KS.f[mmp]-KS.f[ppm])))+(((KS.f[zpp]-KS.f[zmm])+(KS.f[zmp]-KS.f[zpm]))+((KS.f[pzp]-KS.f[mzm])+(KS.f[mzp]-KS.f[pzm])))+(KS.f[zzp]-KS.f[zzm])+KS.fz*n1o2)/KS.rho;
		KS.vx=((((KS.f[ppp]-KS.f[mmm])+(KS.f[pmp]-KS.f[mpm]))+((KS.f[ppm]-KS.f[mmp])+(KS.f[pmm]-KS.f[mpp])))+(((KS.f[pzp]-KS.f[mzm])+(KS.f[pzm]-KS.f[mzp]))+((KS.f[ppz]-KS.f[mmz])+(KS.f[pmz]-KS.f[mpz])))+(KS.f[pzz]-KS.f[mzz])+KS.fx*n1o2)/KS.rho;
		KS.vy=((((KS.f[ppp]-KS.f[mmm])+(KS.f[ppm]-KS.f[mmp]))+((KS.f[mpp]-KS.f[pmm])+(KS.f[mpm]-KS.f[pmp])))+(((KS.f[ppz]-KS.f[mmz])+(KS.f[mpz]-KS.f[pmz]))+((KS.f[zpp]-KS.f[zmm])+(KS.f[zpm]-KS.f[zmp])))+(KS.f[zpz]-KS.f[zmz])+KS.fy*n1o2)/KS.rho;
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void computeDensityAndVelocity_Wall(LBM_KS &KS)
	{
		KS.rho = no1;
		KS.vx = no0;
		KS.vy = no0;
		KS.vz = no0;
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void setEquilibrium(LBM_KS &KS)
	{
		KS.f[mmm] = EQ::eq_mmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mmz] = EQ::eq_mmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mmp] = EQ::eq_mmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mzm] = EQ::eq_mzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mzz] = EQ::eq_mzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mzp] = EQ::eq_mzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mpm] = EQ::eq_mpm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mpz] = EQ::eq_mpz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mpp] = EQ::eq_mpp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zmm] = EQ::eq_zmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zmz] = EQ::eq_zmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zmp] = EQ::eq_zmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zzm] = EQ::eq_zzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zzz] = EQ::eq_zzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zzp] = EQ::eq_zzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zpm] = EQ::eq_zpm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zpz] = EQ::eq_zpz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zpp] = EQ::eq_zpp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pmm] = EQ::eq_pmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pmz] = EQ::eq_pmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pmp] = EQ::eq_pmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pzm] = EQ::eq_pzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pzz] = EQ::eq_pzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pzp] = EQ::eq_pzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[ppm] = EQ::eq_ppm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[ppz] = EQ::eq_ppz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[ppp] = EQ::eq_ppp(KS.rho,KS.vx,KS.vy,KS.vz);
	}

	template< typename LAT_DFS >
	CUDA_HOSTDEV static void setEquilibriumLat(LAT_DFS& f, idx x, idx y, idx z, real rho, real vx, real vy, real vz)
	{
		f(mmm,x,y,z) = EQ::eq_mmm(rho,vx,vy,vz);
		f(zmm,x,y,z) = EQ::eq_zmm(rho,vx,vy,vz);
		f(pmm,x,y,z) = EQ::eq_pmm(rho,vx,vy,vz);
		f(mzm,x,y,z) = EQ::eq_mzm(rho,vx,vy,vz);
		f(zzm,x,y,z) = EQ::eq_zzm(rho,vx,vy,vz);
		f(pzm,x,y,z) = EQ::eq_pzm(rho,vx,vy,vz);
		f(mpm,x,y,z) = EQ::eq_mpm(rho,vx,vy,vz);
		f(zpm,x,y,z) = EQ::eq_zpm(rho,vx,vy,vz);
		f(ppm,x,y,z) = EQ::eq_ppm(rho,vx,vy,vz);

		f(mmz,x,y,z) = EQ::eq_mmz(rho,vx,vy,vz);
		f(zmz,x,y,z) = EQ::eq_zmz(rho,vx,vy,vz);
		f(pmz,x,y,z) = EQ::eq_pmz(rho,vx,vy,vz);
		f(mzz,x,y,z) = EQ::eq_mzz(rho,vx,vy,vz);
		f(zzz,x,y,z) = EQ::eq_zzz(rho,vx,vy,vz);
		f(pzz,x,y,z) = EQ::eq_pzz(rho,vx,vy,vz);
		f(mpz,x,y,z) = EQ::eq_mpz(rho,vx,vy,vz);
		f(zpz,x,y,z) = EQ::eq_zpz(rho,vx,vy,vz);
		f(ppz,x,y,z) = EQ::eq_ppz(rho,vx,vy,vz);

		f(mmp,x,y,z) = EQ::eq_mmp(rho,vx,vy,vz);
		f(zmp,x,y,z) = EQ::eq_zmp(rho,vx,vy,vz);
		f(pmp,x,y,z) = EQ::eq_pmp(rho,vx,vy,vz);
		f(mzp,x,y,z) = EQ::eq_mzp(rho,vx,vy,vz);
		f(zzp,x,y,z) = EQ::eq_zzp(rho,vx,vy,vz);
		f(pzp,x,y,z) = EQ::eq_pzp(rho,vx,vy,vz);
		f(mpp,x,y,z) = EQ::eq_mpp(rho,vx,vy,vz);
		f(zpp,x,y,z) = EQ::eq_zpp(rho,vx,vy,vz);
		f(ppp,x,y,z) = EQ::eq_ppp(rho,vx,vy,vz);
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyDFcur2KS(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		for (int i=0;i<27;i++) KS.f[i] = SD.df(df_cur,i,x,y,z);
	}
};
