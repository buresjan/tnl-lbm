#pragma once

// entropic equilibrium (for KBC) by Robert Straka
//Entropic equilibrium -- somebody should check *FIXME* if this type of equilibrium gives correct 3rd order moment e.g. M^eq_111=rho*u_x*u_y*u_z as
// this is not the case for standard quadratic equilibria! check papers by P.J. Dellar for further details how to handle this cubic defect.

#define _efeq_mmm(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_mmz(vx,vy,vz) (n1o6*n1o6*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy)))
#define _efeq_mmp(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_mzm(vx,vy,vz) (n1o6*n2o3*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_mzz(vx,vy,vz) (n1o6*n2o3*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx)))
#define _efeq_mzp(vx,vy,vz) (n1o6*n2o3*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_mpm(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_mpz(vx,vy,vz) (n1o6*n1o6*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy)))
#define _efeq_mpp(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_zmm(vx,vy,vz) (n2o3*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_zmz(vx,vy,vz) (n2o3*n1o6*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy)))
#define _efeq_zmp(vx,vy,vz) (n2o3*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_zzm(vx,vy,vz) (n2o3*n2o3*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_zzz(vx,vy,vz) (n2o3*n2o3*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz)))
#define _efeq_zzp(vx,vy,vz) (n2o3*n2o3*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_zpm(vx,vy,vz) (n2o3*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_zpz(vx,vy,vz) (n2o3*n1o6*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy)))
#define _efeq_zpp(vx,vy,vz) (n2o3*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_pmm(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_pmz(vx,vy,vz) (n1o6*n1o6*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy)))
#define _efeq_pmp(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_pzm(vx,vy,vz) (n1o6*n2o3*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_pzz(vx,vy,vz) (n1o6*n2o3*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx)))
#define _efeq_pzp(vx,vy,vz) (n1o6*n2o3*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_ppm(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*no1/((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))
#define _efeq_ppz(vx,vy,vz) (n1o6*n1o6*n2o3*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy)))
#define _efeq_ppp(vx,vy,vz) (n1o6*n1o6*n1o6*(no2 - sqrt(no1 + no3*vx*vx))*(no2 - sqrt(no1 + no3*vy*vy))*(no2 - sqrt(no1 + no3*vz*vz))*((no2*vx + sqrt(no1 + no3*vx*vx))/(no1 - vx))*((no2*vy + sqrt(no1 + no3*vy*vy))/(no1 - vy))*((no2*vz + sqrt(no1 + no3*vz*vz))/(no1 - vz)))


template < typename TRAITS >
struct D3Q27_EQ_ENTROPIC
{
	using dreal = typename TRAITS::dreal;

	CUDA_HOSTDEV static dreal eq_zzz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zzz(vx, vy, vz);}

	CUDA_HOSTDEV static dreal eq_pzz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_pzz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mzz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mzz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zpz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zpz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zmz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zmz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zzp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zzp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zzm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zzm(vx, vy, vz);}

	CUDA_HOSTDEV static dreal eq_ppz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_ppz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pmz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_pmz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mpz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mpz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mmz(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mmz(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pzp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_pzp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mzm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mzm(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pzm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_pzm(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mzp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mzp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zpp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zpp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zpm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zpm(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zmp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zmp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_zmm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_zmm(vx, vy, vz);}

	CUDA_HOSTDEV static dreal eq_ppp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_ppp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mmm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mmm(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_ppm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_ppm(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pmp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_pmp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mpp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mpp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mpm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mpm(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_mmp(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_mmp(vx, vy, vz);}
	CUDA_HOSTDEV static dreal eq_pmm(dreal rho, dreal vx, dreal vy, dreal vz) {return rho*_efeq_pmm(vx, vy, vz);}
};
