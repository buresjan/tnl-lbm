#pragma once

#include <TNL/Matrices/MatrixWriter.h>

#include "lbm_common/timeutils.h"
#include "lagrange_3D.h"
#include <TNL/Containers/Vector.h>
#include <complex>
#include <TNL/Matrices/MatrixWriter.h>

template< typename LBM >
auto Lagrange3D<LBM>::hmacroVector(int macro_idx) -> hVectorView
{
	if (macro_idx >= MACRO::N)
		throw std::logic_error("macro_idx must be less than MACRO::N");

	auto& block = lbm.blocks.front();

	// local size of the distributed array
	// FIXME: overlaps
	const idx n = block.local.x() * block.local.y() * block.local.z();

	// offset for the requested quantity
	// FIXME: overlaps
	const idx quantity_offset = block.hmacro.getStorageIndex(macro_idx, block.offset.x(), block.offset.y(), block.offset.z());

	// pointer to the data
	dreal* data = block.hmacro.getData() + quantity_offset;

	TNL_ASSERT_EQ(data, &block.hmacro(macro_idx, block.offset.x(), block.offset.y(), block.offset.z()), "indexing bug");

	return {data, n};
}

template< typename LBM >
auto Lagrange3D<LBM>::dmacroVector(int macro_idx) -> dVectorView
{
	if (macro_idx >= MACRO::N)
		throw std::logic_error("macro_idx must be less than MACRO::N");

	auto& block = lbm.blocks.front();

	// local size of the distributed array
	// FIXME: overlaps
	const idx n = block.local.x() * block.local.y() * block.local.z();

	// offset for the requested quantity
	// FIXME: overlaps
	const idx quantity_offset = block.dmacro.getStorageIndex(macro_idx, block.offset.x(), block.offset.y(), block.offset.z());

	// pointer to the data
	dreal* data = block.dmacro.getData() + quantity_offset;

	return {data, n};
}

template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::computeMinDist()
{
	minDist=1e10;
	maxDist=-1e10;
	for (std::size_t i=0;i<LL.size()-1;i++)
	{
	for (std::size_t j=i+1;j<LL.size();j++)
	{
		real d = dist(LL[j],LL[i]);
		if (d>maxDist) maxDist=d;
		if (d<minDist) minDist=d;
	}
	if (i%1000==0)
	fmt::print("computeMinDist {} of {}\n", i, LL.size());
	}
	return minDist;
}

template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::computeMaxDistFromMinDist(typename LBM::TRAITS::real mindist)
{
	maxDist=-1e10;
	for (std::size_t i=0;i<LL.size()-1;i++)
	{
		real search_dist = 2.0*mindist;
		for (std::size_t j=i+1;j<LL.size();j++)
		{
			real d = dist(LL[j],LL[i]);
			if (d < search_dist)
				if (d>maxDist) maxDist=d;
		}
	}
	return maxDist;
}


template< typename LBM >
void Lagrange3D<LBM>::computeMaxMinDist()
{
	maxDist=-1e10;
	minDist=1e10;
	if (lag_X<=0 || lag_Y<=0) return;
	for (int i=0;i<lag_X-1;i++)
	for (int j=0;j<lag_Y-1;j++)
	{
		int index = findIndex(i,j);
		for (int i1=0;i1<=1;i1++)
		for (int j1=0;j1<=1;j1++)
		if (j1!=0 || i1!=0)
		{
			int index1 = findIndex(i+i1,j+j1);
//			int index1 = findIndex(i+i1,j);
			real d = dist(LL[index1],LL[index]);
			if (d>maxDist) maxDist=d;
			if (d<minDist) minDist=d;
		}
	}
}


template< typename LBM >
int Lagrange3D<LBM>::createIndexArray()
{
	if (lag_X<=0 || lag_Y<=0) return 0;
	index_array = new int*[lag_X];
	for (int i=0;i<lag_X;i++) index_array[i] = new int[lag_Y];
	for (std::size_t k=0;k<LL.size();k++) index_array[LL[k].lag_x][LL[k].lag_y] = k;
	indexed=true;
	return 1;
}


template< typename LBM >
int Lagrange3D<LBM>::findIndex(int i, int j)
{
	if (!indexed) createIndexArray();
	if (!indexed) return 0;
	return index_array[i][j];
	// brute force
//	for (std::size_t k=0;k<LL.size();k++) if (LL[k].lag_x == i && LL[k].lag_y == j) return k;
//	fmt::print("findIndex({},{}): not found\n",i,j);
//	return 0;
}

template< typename LBM >
int Lagrange3D<LBM>::findIndexOfNearestX(typename LBM::TRAITS::real x)
{
	int imin=0;
	real xmindist=fabs(LL[imin].x-x);
	// brute force
	for (std::size_t k=1;k<LL.size();k++)
	if (fabs(LL[k].x - x) < xmindist)
	{
		imin=k;
		xmindist = fabs(LL[imin].x-x);
	}
	return imin;
}
template< typename LBM >
bool Lagrange3D<LBM>::isDDNonZero(int i, real r)
{
	switch (i)
	{
		case 1: // VU: phi3
			if(fabs(r) < 1.0)
				return true;
			else
				return false;
		case 2: // VU: phi2
			if(fabs(r) < 2.0)
				return true;
			else
				return false;;
		case 3: // VU: phi1
			if (fabs(r)>2.0)
				return false;
			else
				return true;
		case 4: // VU: phi4
			if (fabs(r)>1.5)
				return false;
			else
				return true;
	}
}
template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::diracDelta(int i, typename LBM::TRAITS::real r)
{
	if(!isDDNonZero(i, r))
	{
		fmt::print("warning: zero Dirac delta: type={}\n", i);
		return 0;
	}
	else{
		switch (i)
		{
			case 1: // VU: phi3
				return (1.0 - fabs(r));
			case 2: // VU: phi2
				return 0.25*((1.0+cos((PI*r)/2.0)));
			case 3: // VU: phi1
				if (fabs(r)>1.0)
					return (5.0 - 2.0*fabs(r) - sqrt(-7.0 + 12.0*fabs(r) - 4.0*r*r))/8.0;
				else
					return (3.0 - 2.0*fabs(r) + sqrt(1.0 + 4.0*fabs(r) - 4.0*r*r))/8.0;
			case 4: // VU: phi4
				if (fabs(r)>0.5)
					return (5.0 - 3.0*fabs(r) - sqrt(-2.0+6.0*fabs(r)-3.0*r*r))/6.0;
				else
					return (1.0 + sqrt(1.0 - 3.0*r*r))/3.0;
		}
	}
	//Just a backup return if something doesn't return 0
	fmt::print("warning: zero Dirac delta: type={}\n", i);
	return 0;
}
/*
//Commented old Dirac
template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::diracDelta(int i, typename LBM::TRAITS::real r)
{
	switch (i)
	{
		case 1: // VU: phi3
			if(fabs(r) < 1.0)
				return (1.0 - fabs(r));
			else
				return 0;
		case 2: // VU: phi2
			if(fabs(r) < 2.0)
				return 0.25*((1.0+cos((PI*r)/2.0)));
			else
				return 0;
		case 3: // VU: phi1
			if (fabs(r)>2.0)
				return 0;
			else if (fabs(r)>1.0)
				return (5.0 - 2.0*fabs(r) - sqrt(-7.0 + 12.0*fabs(r) - 4.0*r*r))/8.0;
			else
				return (3.0 - 2.0*fabs(r) + sqrt(1.0 + 4.0*fabs(r) - 4.0*r*r))/8.0;
/\*
			if(r > -2.0 && r <= -1.0 )
				result = (5.0 + 2.0*r - sqrt(-7.0 - 12.0*r - 4.0*r*r))/8.0;
			else if(r > -1.0 && r<= 0)
				result = (3.0 + 2.0*r + sqrt(1.0 - 4.0*r - 4.0*r*r))/8.0;
			else if(r > 0 && r <= 1.0)
				result = (3.0 - 2.0*r + sqrt(1.0 + 4.0*r - 4.0*r*r))/8.0;
			else if(r > 1.0 && r <2.0)
				result = (5.0 - 2.0*r - sqrt(-7.0 + 12.0*r - 4.0*r*r))/8.0;
			else result = 0;
*\/
		case 4: // VU: phi4
			if (fabs(r)>1.5)
				return 0;
			else if (fabs(r)>0.5)
				return (5.0 - 3.0*fabs(r) - sqrt(-2.0+6.0*fabs(r)-3.0*r*r))/6.0;
			else
				return (1.0 + sqrt(1.0 - 3.0*r*r))/3.0;
	}
	fmt::print("warning: zero Dirac delta: type={}\n", i);
	return 0;
}
*/

template< typename LBM >
void Lagrange3D<LBM>::constructWuShuMatricesSparse()
{

	if (ws_constructed) return;
	ws_constructed=true;


	int rDirac=1;
	// count non zero elements in matrix ws_A
	int m=LL.size();	// number of lagrangian nodes
	int n=lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();	// number of eulerian nodes
	// projdi veskery filament a najdi sousedni body (ve vzdaelenosti mensi nez je prekryv delta funkci)
	struct DD_struct
	{
		real DD;
		int ka;
	};


	typedef std::vector<DD_struct> VECDD;
	typedef std::vector<int> VEC;
	//typedef std::vector<real> VECR;
	VEC *v = new VEC[m];
	VECDD *vr = new VECDD[m];


	fmt::print("wushu construct loop 1: start\n");
//	int resrv = 2*((ws_speedUpAllocation) ? ws_speedUpAllocationSupport : 10); // we have plenty of memory, we need performance
	real *LLx = new real[m];
	real *LLy = new real[m];
	real *LLz = new real[m];
	int *LLlagx = new int[m];
	for (int i=0;i<m;i++)
	{
		LLx[i]=LL[i].x;
		LLy[i]=LL[i].y;
		LLz[i]=LL[i].z;
		LLlagx[i]=LL[i].lag_x;
	}

	fmt::print("wushu construct loop 1: cont\n");
	#pragma omp parallel for schedule(dynamic)
	for (int el=0;el<m;el++)
	{
		if (el%100==0)
			fmt::print("progress {:5.2f} %    \r", 100.0*el/(real)m);
		for (int ka=0;ka<m;ka++)
		{
			bool proceed=true;
			real d1,d2,ddd;
			if (ws_speedUpAllocation)
			{
				if (abs(LLlagx[el] - LLlagx[ka]) > ws_speedUpAllocationSupport) proceed=false;
			}
			if (!proceed) continue;
			if (ws_regularDirac)
			{
				d1 = diracDelta(rDirac,(LLx[el] - LLx[ka])/lbm.lat.physDl);
				if (d1>0)
				{
					d2 = diracDelta(rDirac, (LLy[el] - LLy[ka])/lbm.lat.physDl);
					if (d2>0)
					{
						ddd=d1*d2*diracDelta(rDirac, (LLz[el] - LLz[ka])/lbm.lat.physDl);
						if (ddd>0)
						{
							v[el].push_back(ka);
							DD_struct sdd;
							sdd.DD = ddd;
							sdd.ka = ka;
							vr[el].push_back(sdd);
						}
					}
				}
			} else
			{
				d1 = diracDelta((LLx[el] - LLx[ka])/lbm.lat.physDl/2.0);
				if (d1>0)
				{
					d2 = diracDelta((LLy[el] - LLy[ka])/lbm.lat.physDl/2.0);
					if (d2>0)
					{
						ddd=d1*d2*diracDelta((LLz[el] - LLz[ka])/lbm.lat.physDl/2.0);
						if (ddd>0)
						{
							v[el].push_back(ka);
							DD_struct sdd;
							sdd.DD = ddd;
							sdd.ka = ka;
							vr[el].push_back(sdd);
						}
					}
				}
			}
		}
	}
	delete [] LLx;
	delete [] LLy;
	delete [] LLz;
	delete [] LLlagx;
	// count non zero
	int nz=0;
	for (int el=0;el<m;el++) nz += vr[el].size();
//	fmt::print("non-zeros: {}\n", nz);

	fmt::print("wushu construct loop 1: end\n");



	// create spmatrix
	ws_A = new SpMatrix<real>();
	ws_A->Solver = Solver;
	ws_A->Ap = new int[m+1];
	ws_A->Ai = new int[nz];
	ws_A->Ax = new real[nz];
	ws_A->m_nz = nz;
	ws_A->m_n = m;

	ws_A->Ap[0]=0;
	for (int i=0;i<nz;i++) ws_A->Ax[i] = 0; // empty

//	fmt::print("Ai construct\n");
	int count=0;
	for (int i=0;i<m;i++)
	{
		ws_A->Ap[i+1] = ws_A->Ap[i] + v[i].size();
//		fmt::print("Ap[{}]={} ({})\n", i+1, ws_A->Ap[i+1], nz);
		for (std::size_t j=0;j<v[i].size();j++)
		{
			ws_A->Ai[count]=v[i][j];
			count++;
		}
	}
	delete [] v;

	// fill vectors delta_el
	// sparse vector of deltas
	d_i = new std::vector<idx>[m];
	d_x = new std::vector<real>[m];
	// fill only non zero elements-relevant
/*
	 // brute force
	fmt::print("wushu construct loop 2: start\n");
	#pragma omp parallel for schedule(static)
	for (int i=0;i<m;i++)
	{
//		if (i%20==0)
//			fmt::print("\r progress: {:04d} of {:04d}", i, m);
		for (int gz=0;gz<lbm.blocks.front().local.z();gz++)
		for (int gy=0;gy<lbm.blocks.front().local.y();gy++)
		for (int gx=0;gx<lbm.blocks.front().local.x();gx++)
		{
			real dd = diracDelta((real)(gx + 0.5) - LL[i].x/lbm.lat.physDl) * diracDelta((real)(gy + 0.5) - LL[i].y/lbm.lat.physDl) * diracDelta((real)(gz + 0.5) - LL[i].z/lbm.lat.physDl);
			if (dd>0)
			{
				d_i[i].push_back(lbm.pos(gx,gy,gz));
				d_x[i].push_back(dd);
			}
		}
	}
	fmt::print("wushu construct loop 2: end\n");
*/

	fmt::print("wushu construct loop 2: start\n");
	idx support=5; // search in this support
	#pragma omp parallel for schedule(static)
	for (int i=0;i<m;i++)
	{
		idx fi_x = floor(LL[i].x/lbm.lat.physDl);
		idx fi_y = floor(LL[i].y/lbm.lat.physDl);
		idx fi_z = floor(LL[i].z/lbm.lat.physDl);

		// FIXME: iterate over LBM blocks
		for (int gz=MAX( 0, fi_z - support);gz<MIN(lbm.blocks.front().local.z(), fi_z + support);gz++)
		for (int gy=MAX( 0, fi_y - support);gy<MIN(lbm.blocks.front().local.y(), fi_y + support);gy++)
		for (int gx=MAX( 0, fi_x - support);gx<MIN(lbm.blocks.front().local.x(), fi_x + support);gx++)
		{
			real dd = diracDelta((real)(gx + 0.5) - LL[i].x/lbm.lat.physDl) * diracDelta((real)(gy + 0.5) - LL[i].y/lbm.lat.physDl) * diracDelta((real)(gz + 0.5) - LL[i].z/lbm.lat.physDl);
			if (dd>0)
			{
				// FIXME: local vs global indices
				d_i[i].push_back(lbm.blocks.front().hmap.getStorageIndex(gx,gy,gz));
				d_x[i].push_back(dd);
			}
		}
	}
	fmt::print("wushu construct loop 2: end\n");


	fmt::print("wushu construct loop 3: start\n");
	#pragma omp parallel for schedule(static)
	for (int i=0;i<m;i++)
	{
		if (i%100==0)
			fmt::print("progress {:5.2f} %    \r", 100.0*i/(real)m);
		for (std::size_t ka=0;ka<vr[i].size();ka++)
		{
			int j=vr[i][ka].ka;
			real ddd = vr[i][ka].DD;
			if (ws_regularDirac)
			{
				ws_A->get(i,j) = ddd;
			} else
			{
				if (ddd>0) // we have non-zero element at i,j
				{
					real val=0;
					for (std::size_t in1=0;in1<d_i[i].size();in1++)
					{
						for (std::size_t in2=0;in2<d_i[j].size();in2++)
						{
							if (d_i[i][in1]==d_i[j][in2])
							{
								val += d_x[i][in1]*d_x[j][in2];
								break;
							}
						}
					}
					ws_A->get(i,j) = val;
				}
			}
		}
	}
	delete [] vr; // free
	fmt::print("wushu construct loop 3: end\n");

	fmt::print("wushu construct loop 4: start\n");
	for (int k=0;k<3;k++)
	{
		ws_x[k] = new real[m];
		ws_b[k] = new real[m]; // right hand side

		ws_hx[k] = new dreal[m];
		ws_hb[k] = new dreal[m]; // right hand side
	}

//	ws_ds = new real[m]; // delta s_ell
	#ifdef USE_CUDA
	for (int k=0;k<3;k++)
	{
		cudaMalloc((void **)&ws_dx[k], m*sizeof(dreal));
		cudaMalloc((void **)&ws_db[k], m*sizeof(dreal));
	}
	cudaMalloc((void **)&ws_du,  n*sizeof(dreal));

	// copy zero to x1, x2 (init)
	dreal* zero = new dreal[m];
	for (int i=0;i<m;i++) zero[i]=0;
	for (int k=0;k<3;k++) cudaMemcpy(ws_dx[k], zero, m*sizeof(dreal), cudaMemcpyHostToDevice); // TODO use setCudaValue ...
	delete [] zero;
	#endif

	// create Matrix M: matrix realizing projection of u* to lagrange desc.
	nz=0;
	for (int el=0;el<m;el++)
	for (std::size_t in1=0;in1<d_i[el].size();in1++)
		nz++;
}
template< typename LBM >
typename Lagrange3D<LBM>::real Lagrange3D<LBM>::calculate3Dirac(int rDirac, int colIndex, int rowIndex, float divisionModifier)
{
	//ka = colIndex
	//el = rowIndex
	real d1; //dirac 1
	real d2; //dirac 2
	real d3; //dirac 3
	real ddd;

				d1 = diracDelta(rDirac,(LL[rowIndex].x - LL[colIndex].x)/lbm.lat.physDl/divisionModifier);
				if (d1>0)
				{
					d2 = diracDelta(rDirac, (LL[rowIndex].y - LL[colIndex].y)/lbm.lat.physDl/divisionModifier);
					if (d2>0)
					{
						d3=diracDelta(rDirac, (LL[rowIndex].z - LL[colIndex].z)/lbm.lat.physDl/divisionModifier);
						if (d3>0)
						{
							ddd = d1*d2*d3;
							//fmt::print("Dirac result: {}  \n", ddd);
							return ddd;
						}

					}

				}
				return 0;
}

template< typename LBM >
bool Lagrange3D<LBM>::is3DiracNonZero(int rDirac, int colIndex, int rowIndex, float divisionModifier)
{
	//ka = colIndex
	//el = rowIndex
	bool d1; //dirac 1
	bool d2; //dirac 2
	bool d3; //dirac 3

	d1 = isDDNonZero(rDirac,(LL[rowIndex].x - LL[colIndex].x)/lbm.lat.physDl/divisionModifier);
	if (d1)
	{
		d2 = isDDNonZero(rDirac, (LL[rowIndex].y - LL[colIndex].y)/lbm.lat.physDl/divisionModifier);
		if (d2)
		{
			d3=isDDNonZero(rDirac, (LL[rowIndex].z - LL[colIndex].z)/lbm.lat.physDl/divisionModifier);
			if (d3)
			{
				return true;
			}
		}
	}
	return false;
}

template< typename LBM >
void Lagrange3D<LBM>::constructWuShuMatricesSparse_TNL()
{
	if (ws_tnl_constructed) return;
	ws_tnl_constructed=true;
	int rDirac=1;
	// count non zero elements in matrix A
	int m=LL.size();	// number of lagrangian nodes
	int n=lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();	// number of eulerian nodes
	// projdi veskery filament a najdi sousedni body (ve vzdaelenosti mensi nez je prekryv delta funkci)
	struct DD_struct
	{
		real DD;
		int ka; //col index
	};
	typedef std::vector<DD_struct> VECDD;
	typedef std::vector<int> VEC;
	typedef std::vector<real> VECR;

	//VEC *v = new VEC[m];
	VECDD *vr = new VECDD[m];
	int *elementCounts = new int[m];

	fmt::print("tnl wushu construct loop 1: start\n");

	fmt::print("tnl wushu construct loop 1: cont\n");

	//TODO: look into OMP parallelization to avoid issues
	#pragma omp parallel for schedule(dynamic)
	//This could cause issues ^^
	//Test for CPU
	//TODO: Rename index
	//EL = Row index
	//KA = Column index
	//ddd = matrix value

	for (int el=0;el<m;el++)
	{
		int rowCapacity = 0;  //Number of elements where DiracDelta > 0
		if (el%100==0)
			fmt::print("progress {:5.2f} %    \r", 100.0*el/(real)m);
		for (int ka=0;ka<m;ka++)
		{
			real ddd;

			if (ws_regularDirac)
			{
				//calculate dirac with selected dirac type
				ddd = calculate3Dirac(rDirac, ka, el);
				if(is3DiracNonZero(rDirac, ka, el))
				{
					rowCapacity++;
				}

			} else
			{
				//calculate ddd with default dirac type
				ddd = calculate3Dirac(diracDeltaType, ka, el, 2.0);
				if(is3DiracNonZero(rDirac, ka, el, 2.0))
				{
					rowCapacity++;
				}
			}
			if(ddd>0)
			{
					DD_struct sdd;
					sdd.DD = ddd;
					sdd.ka = ka;
					vr[el].push_back(sdd);
			}
		}
		elementCounts[el] = rowCapacity;
	}


	fmt::print("tnl wushu construct loop 1: end\n");

	// allocate matrix A
	ws_tnl_hA = std::make_shared< hEllpack >();
	ws_tnl_hA->setDimensions(m, m);
//	int max_nz_per_row=0;
//	for (int el=0;el<m;el++) {
//		max_nz_per_row = TNL::max(max_nz_per_row, vr[el].size());
//	}
//	ws_tnl_hA->setConstantCompressedRowLengths(max_nz_per_row);
//	typename hEllpack::RowCapacitiesType hA_row_lengths( m );
//	for (int el=0; el<m; el++) hA_row_lengths[el] = vr[el].size();
//	ws_tnl_hA->setRowCapacities(hA_row_lengths);
	typename hEllpack::RowCapacitiesType hA_row_lengths( m );
	for (int el=0; el<m; el++) hA_row_lengths[el] = elementCounts[el];
	ws_tnl_hA->setRowCapacities(hA_row_lengths);



	// fill vectors delta_el
	// sparse vector of deltas
	//TODO: This could be rewritten with DDStruct
	d_i = new std::vector<idx>[m];
	d_x = new  std::vector<real>[m];
	// fill only non zero elements-relevant

	fmt::print("tnl wushu construct loop 2: start\n");
	idx support=5; // search in this support
	#pragma omp parallel for schedule(static)
	for (int i=0;i<m;i++)
	{
		idx fi_x = floor(LL[i].x/lbm.lat.physDl);
		idx fi_y = floor(LL[i].y/lbm.lat.physDl);
		idx fi_z = floor(LL[i].z/lbm.lat.physDl);

		// FIXME: iterate over LBM blocks
		for (int gz=MAX( 0, fi_z - support);gz<MIN(lbm.blocks.front().local.z(), fi_z + support);gz++)
		for (int gy=MAX( 0, fi_y - support);gy<MIN(lbm.blocks.front().local.y(), fi_y + support);gy++)
		for (int gx=MAX( 0, fi_x - support);gx<MIN(lbm.blocks.front().local.x(), fi_x + support);gx++)
		{
			real dd = diracDelta((real)(gx + 0.5) - LL[i].x/lbm.lat.physDl) * diracDelta((real)(gy + 0.5) - LL[i].y/lbm.lat.physDl) * diracDelta((real)(gz + 0.5) - LL[i].z/lbm.lat.physDl);
			if (dd>0)
			{
				// FIXME: local vs global indices
				d_i[i].push_back(lbm.blocks.front().hmap.getStorageIndex(gx,gy,gz));
				d_x[i].push_back(dd);
			}
		}
	}
	fmt::print("tnl wushu construct loop 2: end\n");

	fmt::print("tnl wushu construct loop 3: start\n");
	#pragma omp parallel for schedule(static)
	for (int i=0;i<m;i++)
	{
		if (i%100==0)
			fmt::print("progress {:5.2f} %    \r", 100.0*i/(real)m);
		for (std::size_t ka=0;ka<elementCounts[i];ka++)
		{
			int j=vr[i][ka].ka;
			real ddd = vr[i][ka].DD;
			if (ws_regularDirac)
			{
				ws_tnl_hA->setElement(i,j, ddd);
			} else
			{
				real val=0;
				for (std::size_t in1=0;in1<d_i[i].size();in1++)
				{
					for (std::size_t in2=0;in2<d_i[j].size();in2++)
					{
						if (d_i[i][in1]==d_i[j][in2])
						{
							val += d_x[i][in1]*d_x[j][in2];
							break;
						}
					}
				}
				ws_tnl_hA->setElement(i,j, val);
			}
		}
	}
	delete [] vr; // free
	fmt::print("tnl wushu construct loop 3: end\n");

	// create vectors for the solution of the linear system
	for (int k=0;k<3;k++)
	{
		ws_tnl_hx[k].setSize(m);
		ws_tnl_hb[k].setSize(m);
		#ifdef USE_CUDA
		ws_tnl_dx[k].setSize(m);
		ws_tnl_db[k].setSize(m);
		ws_tnl_hxz[k].setSize(m);
		ws_tnl_hbz[k].setSize(m);
		#endif
	}

	// zero-initialize x1, x2, x3
	for (int k=0;k<3;k++) ws_tnl_hx[k].setValue(0);
	#ifdef USE_CUDA
	for (int k=0;k<3;k++) ws_tnl_dx[k].setValue(0);
	for (int k=0;k<3;k++) ws_tnl_hxz[k].setValue(0);
	#endif

	#ifdef USE_CUDA
	// create Matrix M: matrix realizing projection of u* to lagrange desc.
	ws_tnl_hM.setDimensions(m, n);
//	max_nz_per_row = 0;
//	for (int el=0;el<m;el++)
//		max_nz_per_row = TNL::max(max_nz_per_row, d_i[el].size());
//	ws_tnl_hM.setConstantCompressedRowLengths(max_nz_per_row);
	typename hEllpack::RowCapacitiesType hM_row_lengths( m );
	for (int el=0; el<m; el++) hM_row_lengths[el] = d_i[el].size();
	ws_tnl_hM.setRowCapacities(hM_row_lengths);

//	fmt::print("Ai construct\n");
	for (int i=0;i<m;i++)
	{
		auto row = ws_tnl_hM.getRow(i);
		for (std::size_t j=0;j<d_i[i].size();j++)
			row.setElement(j, d_i[i][j], (dreal)d_x[i][j]);
	}

	// its transpose
	ws_tnl_hMT.setDimensions(n, m);

	// for each Euler node, assign
	VEC *vn = new VEC[n];
	VECR *vx = new VECR[n];
	for (int i=0;i<m;i++)
	for (std::size_t j=0;j<d_i[i].size();j++)
	{
		vn[ d_i[i][j] ].push_back( i );
		vx[ d_i[i][j] ].push_back( d_x[i][j] );
	}

//	max_nz_per_row = 0;
//	for (int el=0;el<n;el++)
//		max_nz_per_row = TNL::max(max_nz_per_row, vn[el].size());
//	ws_tnl_hMT.setConstantCompressedRowLengths(max_nz_per_row);
	typename hEllpack::RowCapacitiesType hMT_row_lengths( n );
	for (int el=0; el<n; el++) hMT_row_lengths[el] = vn[el].size();
	ws_tnl_hMT.setRowCapacities(hMT_row_lengths);

	for (int i=0;i<n;i++)
	{
		auto row = ws_tnl_hMT.getRow(i);
		for (std::size_t j=0;j<vn[i].size();j++)
			row.setElement(j, vn[i][j], vx[i][j]);
	}
	delete [] vn;
	delete [] vx;
	#endif

	// output to files
	TNL::Matrices::MatrixWriter< hEllpack >::writeMtx( "ws_tnl_hA.mtx", *ws_tnl_hA );

	// update the preconditioner
	ws_tnl_hprecond->update(ws_tnl_hA);
	ws_tnl_hsolver.setMatrix(ws_tnl_hA);

	#ifdef USE_CUDA
	// copy matrices from host to the GPU
	ws_tnl_dA = std::make_shared< dEllpack >();
	*ws_tnl_dA = *ws_tnl_hA;
	ws_tnl_dM = ws_tnl_hM;
	ws_tnl_dMT = ws_tnl_hMT;

	// update the preconditioner
	ws_tnl_dprecond->update(ws_tnl_dA);
	ws_tnl_dsolver.setMatrix(ws_tnl_dA);

	TNL::Matrices::MatrixWriter< hEllpack >::writeMtx( "original_matrices/ws_tnl_hM_original_method-"+std::to_string(!ws_regularDirac)+"_dirac-"+std::to_string(diracDeltaType)+".mtx", ws_tnl_hM );
	TNL::Matrices::MatrixWriter< hEllpack >::writeMtx( "original_matrices/ws_tnl_hA_original_method-"+std::to_string(!ws_regularDirac)+"_dirac-"+std::to_string(diracDeltaType)+".mtx", *ws_tnl_hA );

	#endif
	fmt::print("tnl wushu lagrange_3D_end\n");
	fmt::print("number of lagrangian points: {}\n", m);

	const char* compute_desc = "undefined";
	switch (ws_compute)
	{
		case ws_computeCPU:                    compute_desc = "ws_computeCPU"; break;
		case ws_computeCPU_TNL:                compute_desc = "ws_computeCPU_TNL"; break;
		case ws_computeGPU_TNL:                compute_desc = "ws_computeGPU_TNL"; break;
		case ws_computeHybrid_TNL:             compute_desc = "ws_computeHybrid_TNL"; break;
		case ws_computeHybrid_TNL_zerocopy:    compute_desc = "ws_computeHybrid_TNL_zerocopy"; break;
	}
	log("constructed WuShu matrices for ws_compute={}", compute_desc);
}

template< typename Matrix, typename Vector >
__cuda_callable__
typename Matrix::RealType
rowVectorProduct( const Matrix& matrix, typename Matrix::IndexType i, const Vector& vector )
{
    typename Matrix::RealType result = 0;
    const auto row = matrix.getRow( i );

    for( typename Matrix::IndexType c = 0; c < row.getSize(); c++ ) {
        const typename Matrix::IndexType column = row.getColumnIndex( c );
        if( column != TNL::Matrices::paddingIndex< typename Matrix::IndexType > )
            result += row.getValue( c ) * vector[ column ];
    }

    return result;
}

//require: rho, vx, vy, vz
template< typename LBM >
void Lagrange3D<LBM>::computeWuShuForcesSparse(real time)
{
	switch (ws_compute)
	{
		case ws_computeCPU_TNL:
		case ws_computeGPU_TNL:
		case ws_computeHybrid_TNL:
		case ws_computeHybrid_TNL_zerocopy:
			constructWuShuMatricesSparse_TNL();
			break;
		default:
			constructWuShuMatricesSparse();
			break;
	}

	int m=LL.size();
	idx n=lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();

	const auto drho = dmacroVector(MACRO::e_rho);
	const auto dvx = dmacroVector(MACRO::e_vx);
	const auto dvy = dmacroVector(MACRO::e_vy);
	const auto dvz = dmacroVector(MACRO::e_vz);
	auto dfx = dmacroVector(MACRO::e_fx);
	auto dfy = dmacroVector(MACRO::e_fy);
	auto dfz = dmacroVector(MACRO::e_fz);

	const auto hrho = hmacroVector(MACRO::e_rho);
	const auto hvx = hmacroVector(MACRO::e_vx);
	const auto hvy = hmacroVector(MACRO::e_vy);
	const auto hvz = hmacroVector(MACRO::e_vz);
	auto hfx = hmacroVector(MACRO::e_fx);
	auto hfy = hmacroVector(MACRO::e_fy);
	auto hfz = hmacroVector(MACRO::e_fz);

	switch (ws_compute)
	{
		#ifdef USE_CUDA
		case ws_computeGPU_TNL:
		{
			// no Device--Host copy is required
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_db[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_db[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_db[2], -1.0);
			// solver
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_dsolver.solve(ws_tnl_db[k], ws_tnl_dx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_dsolver.getIterations(), ws_tnl_dsolver.getResidue());
			}
			const auto x1 = ws_tnl_dx[0].getConstView();
			const auto x2 = ws_tnl_dx[1].getConstView();
			const auto x3 = ws_tnl_dx[2].getConstView();
			//TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}

		case ws_computeHybrid_TNL:
		{
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_db[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_db[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_db[2], -1.0);
			// copy to Host
			for (int k=0;k<3;k++) ws_tnl_hb[k] = ws_tnl_db[k];
			// solve on CPU
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hb[k], ws_tnl_hx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			// copy to GPU
			for (int k=0;k<3;k++) ws_tnl_dx[k] = ws_tnl_hx[k];
			// continue on GPU
			const auto x1 = ws_tnl_dx[0].getConstView();
			const auto x2 = ws_tnl_dx[1].getConstView();
			const auto x3 = ws_tnl_dx[2].getConstView();
//			TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}

		case ws_computeHybrid_TNL_zerocopy:
		{
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_hbz[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_hbz[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_hbz[2], -1.0);
			// solve on CPU
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hbz[k].getView(), ws_tnl_hxz[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			// continue on GPU
			const auto x1 = ws_tnl_hxz[0].getConstView();
			const auto x2 = ws_tnl_hxz[1].getConstView();
			const auto x3 = ws_tnl_hxz[2].getConstView();
//			TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}
		#endif // USE_CUDA
		case ws_computeCPU_TNL:
		{
			// vx, vy, vz, rho must be copied from the device
			ws_tnl_hM.vectorProduct(hvx, ws_tnl_hb[0], -1.0);
			ws_tnl_hM.vectorProduct(hvy, ws_tnl_hb[1], -1.0);
			ws_tnl_hM.vectorProduct(hvz, ws_tnl_hb[2], -1.0);
			// solver
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hb[k], ws_tnl_hx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			auto kernel = [&] (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( ws_tnl_hMT.getRowCapacity(i) > 0 ) {
					hfx[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[0]);
					hfy[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[1]);
					hfz[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[2]);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Host >((idx) 0, n, kernel);
			// copy forces to the device
			// FIXME: this is copied multiple times when there are multiple Lagrange3D objects
			// (ideally there should be only one Lagrange3D object that comprises all immersed bodies)
			dfx = hfx;
			dfy = hfy;
			dfz = hfz;
			break;
		}
		case ws_computeCPU:
		{
			// vx, vy, rho must be copied from the device
			// fx, fy must be zero
			for (int el=0;el<m;el++)
			{
				for (int k=0;k<3;k++)
					ws_b[k][el]=0;
				for (std::size_t in1=0;in1<d_i[el].size();in1++)
				{
					int gi=d_i[el][in1];
					ws_b[0][el] -= hvx[gi] * d_x[el][in1];
					ws_b[1][el] -= hvy[gi] * d_x[el][in1];
					ws_b[2][el] -= hvz[gi] * d_x[el][in1];
				}
			}
			//solver
			for (int k=0;k<3;k++) ws_A->solve(ws_b[k],ws_x[k]);
			// transfer to fx fy
			for (int el=0;el<m;el++)
			{
				for (std::size_t in1=0;in1<d_i[el].size();in1++)
				{
					int gi=d_i[el][in1];
					hfx[gi] += 2 * hrho[gi] * ws_x[0][el]*d_x[el][in1];
					hfy[gi] += 2 * hrho[gi] * ws_x[1][el]*d_x[el][in1];
					hfz[gi] += 2 * hrho[gi] * ws_x[2][el]*d_x[el][in1];
				}
			}
			// copy forces to the device
			// FIXME: this is copied multiple times when there are multiple Lagrange3D objects
			// (ideally there should be only one Lagrange3D object that comprises all immersed bodies)
			dfx = hfx;
			dfy = hfy;
			dfz = hfz;
			break;
		}
		default:
			fmt::print(stderr, "lagrange_3D: Wu Shu compute flag {} unrecognized.\n", ws_compute);
			break;
	}
}

template< typename LBM >
template< typename... ARGS >
void Lagrange3D<LBM>::log(const char* fmts, ARGS... args)
{
	FILE* f = fopen(logfile.c_str(), "at"); // append information
	if (f==0) {
		fmt::print(stderr, "unable to create/access file {}\n", logfile);
		return;
	}
	// insert time stamp
	fmt::print(f, "{} ", timestamp());
	fmt::print(f, fmts, args...);
	fmt::print(f, "\n");
	fclose(f);

	//fmt::print(fmts, args...);
	//fmt::print("\n");
}

template< typename LBM >
Lagrange3D<LBM>::Lagrange3D(LBM &inputLBM, const std::string& resultsDir) : lbm(inputLBM)
{
	logfile = fmt::format("{}/ibm_solver.log", resultsDir);

	ws_tnl_hsolver.setMaxIterations(10000);
	ws_tnl_hsolver.setConvergenceResidue(3e-4);
	ws_tnl_hprecond = std::make_shared< hPreconditioner >();
//	ws_tnl_hsolver.setPreconditioner(ws_tnl_hprecond);
	#ifdef USE_CUDA
	ws_tnl_dsolver.setMaxIterations(10000);
	ws_tnl_dsolver.setConvergenceResidue(3e-4);
	ws_tnl_dprecond = std::make_shared< dPreconditioner >();
//	ws_tnl_dsolver.setPreconditioner(ws_tnl_dprecond);
	#endif

	ws_A=0;
	for (int k=0;k<3;k++)
	{
		ws_x[k]=0;
		ws_b[k]=0;
		ws_hx[k]=0;
		ws_hb[k]=0;
		ws_dx[k]=0;
		ws_db[k]=0;
	}
	ws_du=0;
	d_i=0;
	d_x=0;
}

template< typename LBM >
Lagrange3D<LBM>::~Lagrange3D()
{
	if (index_array)
	{
		for (int i=0;i<lag_X;i++) delete [] index_array[i];
		delete [] index_array;
	}
	// WuShu
	if (d_i) delete [] d_i;
	if (d_x) delete [] d_x;

	// WuShu
	if (ws_constructed)
	{
		for (int k=0;k<3;k++)
		{
			if (ws_x[k]) delete [] ws_x[k];
			if (ws_b[k]) delete [] ws_b[k];
			if (ws_hx[k]) delete [] ws_hx[k];
			if (ws_hb[k]) delete [] ws_hb[k];
		}
		if (ws_A) delete ws_A;
	}
}
