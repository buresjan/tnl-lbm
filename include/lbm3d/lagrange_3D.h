#pragma once

#include <TNL/Devices/Host.h>
#include <vector>
#include <algorithm>
#include <string>

#include <math.h>

#include "defs.h"
#include "lbm.h"

#include <fmt/core.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/Preconditioners/Diagonal.h>
#include <TNL/Solvers/Linear/Preconditioners/ILU0.h>
#include <TNL/Allocators/CudaHost.h>
#include <TNL/Containers/Vector.h>

template< typename Device, typename Index, typename IndexAlocator >
using SlicedEllpackSegments = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAlocator >;
template< typename Real, typename Device, typename Index >
using SlicedEllpack = TNL::Matrices::SparseMatrix< Real,
													Device,
													Index,
													TNL::Matrices::GeneralMatrix,
													SlicedEllpackSegments
													>;

enum {
	ws_computeCPU_TNL,
	ws_computeGPU_TNL,
	ws_computeHybrid_TNL,
	ws_computeHybrid_TNL_zerocopy
};

// cyclic vector
template < typename T >
class CyclicVector : public std::vector<T>
{
public:
    T& operator[](int n)
    {
                if (n<0)
                        return operator[](n+std::vector<T>::size());
                if (std::size_t(n)>=std::vector<T>::size())
                        return operator[](n-std::vector<T>::size());
                return std::vector<T>::operator[](n);
//        int n=in;
//        if (in<0) n=std::vector<T>::size()+in; else
//        if (in>=std::vector<T>::size()) n=in-std::vector<T>::size();
//        return std::vector<T>::operator[](n);
    }
	const T& operator[](int n) const
    {
                if (n<0)
                        return operator[](n+std::vector<T>::size());
                if (std::size_t(n)>=std::vector<T>::size())
                        return operator[](n-std::vector<T>::size());
                return std::vector<T>::operator[](n);
//        int n=in;
//        if (in<0) n=std::vector<T>::size()+in; else
//        if (in>=std::vector<T>::size()) n=in-std::vector<T>::size();
//        return std::vector<T>::operator[](n);
    }
};

template < typename REAL>
struct LagrangePoint3D
{
	using Real = REAL;

	REAL x=0,y=0,z=0;
	// Lagrangian coordinates of the surface (as a grid)
	int lag_x, lag_y;

	CUDA_HOSTDEV LagrangePoint3D<REAL> operator/(REAL r) const
	{
		LagrangePoint3D<REAL> p;
		p.x = x/r;
		p.y = y/r;
		p.z = z/r;
		return p;
	}
	CUDA_HOSTDEV LagrangePoint3D<REAL>&operator/=(REAL r)
	{
		this->x/=r;
		this->y/=r;
		this->z/=r;
		return *this;
	}
	template <typename T>
	CUDA_HOSTDEV LagrangePoint3D<REAL>&operator=(LagrangePoint3D<T> other)
	{
		this->x=(REAL)other.x;
		this->y=(REAL)other.y;
		this->z=(REAL)other.z;
		this->lag_x = other.lag_x;
		this->lag_y = other.lag_y;
		return *this;
	}
};

enum class DiracMethod //Enum for deciding which method is used for calculation
{
	MODIFIED = 0,
	ORIGINAL = 1,
};

template< typename LBM >
struct Lagrange3D
{
	using TRAITS = typename LBM::TRAITS;
	using MACRO = typename LBM::MACRO;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;

	LBM &lbm;

//	using hVector = TNL::Containers::Vector< real, TNL::Devices::Host, idx, TNL::Allocators::CudaHost<real> >;
	using hVector = TNL::Containers::Vector< dreal, TNL::Devices::Host, idx, TNL::Allocators::CudaHost<dreal> >;
	using dVector = TNL::Containers::Vector< dreal, TNL::Devices::Cuda, idx >;
//	using hEllpack = SlicedEllpack< real, TNL::Devices::Host, idx >;
	using hEllpack = SlicedEllpack< dreal, TNL::Devices::Host, idx >;
	using dEllpack = SlicedEllpack< dreal, TNL::Devices::Cuda, idx >;
	using hEllpackPtr = std::shared_ptr< hEllpack >;
	using dEllpackPtr = std::shared_ptr< dEllpack >;

	using hVectorView = TNL::Containers::VectorView< dreal, TNL::Devices::Host, idx >;
	using dVectorView = TNL::Containers::VectorView< dreal, TNL::Devices::Cuda, idx >;

	// ws_ using sparse matrices
	hEllpackPtr ws_tnl_hA;
	hEllpack ws_tnl_hM; // matrix realizing projection of u* to lagrange desc.
	hEllpack ws_tnl_hMT; // matrix realizing projection of uB from lagrange desc. to Euler desc. .... basially transpose of M
	hVector ws_tnl_hx[3], ws_tnl_hb[3];

	using hSolver = TNL::Solvers::Linear::CG< hEllpack >;
	using hPreconditioner = TNL::Solvers::Linear::Preconditioners::Diagonal< hEllpack >;
//	using hPreconditioner = TNL::Solvers::Linear::Preconditioners::ILU0< hEllpack >;
	hSolver ws_tnl_hsolver;
	typename std::shared_ptr< hPreconditioner > ws_tnl_hprecond;

	#ifdef USE_CUDA
	dEllpackPtr ws_tnl_dA; // square matrix A
	dEllpack ws_tnl_dM; // matrix realizing projection of u* to lagrange desc.
	dEllpack ws_tnl_dMT; // matrix realizing projection of uB from lagrange desc. to Euler desc. .... basially transpose of M
	dVector ws_tnl_dx[3], ws_tnl_db[3];
	// for ws_computeHybrid_TNL_zerocopy
	using hVectorPinned = TNL::Containers::Vector< dreal, TNL::Devices::Host, idx, TNL::Allocators::CudaHost<dreal> >;
	hVectorPinned ws_tnl_hxz[3], ws_tnl_hbz[3];

	using dSolver = TNL::Solvers::Linear::CG< dEllpack >;
	using dPreconditioner = TNL::Solvers::Linear::Preconditioners::Diagonal< dEllpack >;
//	using dPreconditioner = TNL::Solvers::Linear::Preconditioners::ILU0< dEllpack >;
	dSolver ws_tnl_dsolver;
	typename std::shared_ptr< dPreconditioner > ws_tnl_dprecond;
	#endif
	bool ws_tnl_constructed=false;

	bool indexed=false; // is i,j lagrangian index created?
	int **index_array=0;

	// Lagrange surface dimensions
	int lag_X=-1; // size
	int lag_Y=-1; // size

	int obj_id = 0;

	DiracMethod methodVariant=DiracMethod::MODIFIED;		// use continuous ws_ trick with 2 dirac functions
	int ws_compute=ws_computeGPU_TNL;		// ws_computeCPU, ws_computeGPU, ws_computeHybrid

	void constructWuShuMatricesSparse_TNL();
	void constructWuShuMatricesSparseGPU_TNL();
	void computeWuShuForcesSparse(real time);

	bool ws_constructed=false;	// Wu Shu matrices constructed?

	real maxDist;			// maximal distance between points
	real minDist;			// minimal distance between points
	int diracDeltaTypeEL=2;
	int diracDeltaTypeLL=1;

	CyclicVector<LagrangePoint3D<real>> LL;

	using DLPVECTOR_REAL = TNL::Containers::Vector<LagrangePoint3D<real>,TNL::Devices::Cuda>;
	//TODO: Change real type here during testing
	using DLPVECTOR_DREAL = TNL::Containers::Vector<LagrangePoint3D<dreal>,TNL::Devices::Cuda>;
	//using DLPVECTOR_DREAL = TNL::Containers::Vector<LagrangePoint3D<real>,TNL::Devices::Cuda>;

	using HLPVECTOR_REAL = TNL::Containers::Vector<LagrangePoint3D<real>,TNL::Devices::Host>;
	//TODO: Change real type here during testing
	using HLPVECTOR_DREAL = TNL::Containers::Vector<LagrangePoint3D<dreal>,TNL::Devices::Host>;
	//using HLPVECTOR_DREAL = TNL::Containers::Vector<LagrangePoint3D<real>,TNL::Devices::Host>;

	// accessors for macroscopic quantities as a 1D vector
	hVectorView hmacroVector(int macro_idx);  // macro_idx must be less than MACRO::N
	dVectorView dmacroVector(int macro_idx);  // macro_idx must be less than MACRO::N

	real dist(LagrangePoint3D<real> &A, LagrangePoint3D<real> &B) { return NORM( A.x - B.x, A.y - B.y, A.z - B.z ); }

	int findIndex(int i, int j);
	int createIndexArray();
	int findIndexOfNearestX(real x);

	void computeMaxMinDist();	// computes min and max distance between neinghboring nodes
	real computeMinDist();		// computes min and max distance between neinghboring nodes
	real computeMaxDistFromMinDist(real mindist);		// computes min and max distance between neighboring nodes
//	void integrateForce(real &Fx, real &Fy, real &Fz, real surface_element_size) { printf("integrateForce not implemented yet."); }

	// special log file for the linear system solvers
	std::string logfile;

	template < typename... ARGS >
	void log(const char* fmt, ARGS... args);

	// constructors
	Lagrange3D(LBM &inputLBM, const std::string& resultsDir, int obj_id);
	~Lagrange3D();

	// disable copy-constructor and copy-assignment, leave only move-constructor and move-assignment
	// (because this class has a "LBM &lbm;" member)
	Lagrange3D(const Lagrange3D&) = delete;
	Lagrange3D(Lagrange3D&&) = default;
	Lagrange3D& operator=(const Lagrange3D&) = delete;
	Lagrange3D& operator=(Lagrange3D&&) = default;
};

#include "lagrange_3D.hpp"
