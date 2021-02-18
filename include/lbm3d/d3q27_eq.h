#pragma once

//#define feq(qx, qy, qz, vx, vy, vz) no1 - n3o2 * ((vx)*(vx) + (vy)*(vy) + (vz)*(vz)) + no3*((qx)*(vx) + (qy)*(vy) + (qz)*(vz)) + n9o2*((qx)*(vx) + (qy)*(vy) + (qz)*(vz))*((qx)*(vx) + (qy)*(vy) + (qz)*(vz))

// second order Maxwell-Boltzmann Equilibrium
template < typename TRAITS >
struct D3Q27_EQ
{
	using dreal = typename TRAITS::dreal;

	CUDA_HOSTDEV_NOINLINE static dreal feq(int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		return no1 - n3o2 * (vx*vx + vy*vy + vz*vz) + no3*(qx*vx + qy*vy + qz*vz) + n9o2*(qx*vx + qy*vy + qz*vz)*(qx*vx + qy*vy + qz*vz);
	}

	CUDA_HOSTDEV static dreal eq_zzz(dreal rho, dreal vx, dreal vy, dreal vz) {return n8o27*rho*feq( 0, 0, 0,  vx, vy, vz);}

	CUDA_HOSTDEV static dreal eq_pzz(dreal rho, dreal vx, dreal vy, dreal vz) {return n2o27*rho*feq( 1, 0, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mzz(dreal rho, dreal vx, dreal vy, dreal vz) {return n2o27*rho*feq(-1, 0, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zpz(dreal rho, dreal vx, dreal vy, dreal vz) {return n2o27*rho*feq( 0, 1, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zmz(dreal rho, dreal vx, dreal vy, dreal vz) {return n2o27*rho*feq( 0,-1, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zzp(dreal rho, dreal vx, dreal vy, dreal vz) {return n2o27*rho*feq( 0, 0, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zzm(dreal rho, dreal vx, dreal vy, dreal vz) {return n2o27*rho*feq( 0, 0,-1, vx, vy, vz);}

	CUDA_HOSTDEV static dreal eq_ppz(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 1, 1, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pmz(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 1,-1, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mpz(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq(-1, 1, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mmz(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq(-1,-1, 0, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pzp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 1, 0, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mzm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq(-1, 0,-1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pzm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 1, 0,-1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mzp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq(-1, 0, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zpp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 0, 1, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zpm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 0, 1,-1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zmp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 0,-1, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zmm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o54*rho*feq( 0,-1,-1, vx, vy, vz);}

	CUDA_HOSTDEV static dreal eq_ppp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq( 1, 1, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mmm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq(-1,-1,-1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_ppm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq( 1, 1,-1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pmp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq( 1,-1, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mpp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq(-1, 1, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mpm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq(-1, 1,-1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mmp(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq(-1,-1, 1, vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pmm(dreal rho, dreal vx, dreal vy, dreal vz) {return n1o216*rho*feq( 1,-1,-1, vx, vy, vz);}
};

//#undef feq
