#pragma once
#include "defs.h"


//TODO: Missing default case
template< typename real >
CUDA_HOSTDEV bool isDDNonZero(int i, real r)
{
	switch (i)
	{
		case 1: // VU: phi3
			if(fabs(r) < (real)1.0)
				return true;
			else
				return false;
		case 2: // VU: phi2
			if(fabs(r) < (real)2.0)
				return true;
			else
				return false;
		case 3: // VU: phi1
			if (fabs(r)>=(real)2.0)
				return false;
			else
				return true;
		case 4: // VU: phi4
			if (fabs(r)>=(real)1.5)
				return false;
			else
				return true;
	}
	return false;
}
//TODO: Convert numbers to (reals)
//TODO: Remove unnecessary isDDNonZero checks
template< typename real >
CUDA_HOSTDEV real diracDelta(int i, real r)
{
	if(!isDDNonZero(i, r))
	{
		//fmt::print("A / warning: zero Dirac delta: type={}\n", i);
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
	//fmt::print("warning: zero Dirac delta: type={}\n", i);
	return 0;
}


template< typename LLVectorType>
CUDA_HOSTDEV bool is3DiracNonZero(int rDirac, int colIndex, int rowIndex,const LLVectorType& LL)
{
	bool d1; //dirac 1
	bool d2; //dirac 2
	bool d3; //dirac 3

	d1 = isDDNonZero(rDirac,(LL[rowIndex].x - LL[colIndex].x));
	if (d1)
	{
		d2 = isDDNonZero(rDirac, (LL[rowIndex].y - LL[colIndex].y));
		if (d2)
		{
			d3=isDDNonZero(rDirac, (LL[rowIndex].z - LL[colIndex].z));
			if (d3)
			{
				return true;
			}
		}
	}
	return false;
}


template< typename LLVectorType >
CUDA_HOSTDEV typename LLVectorType::ValueType::Real calculate3Dirac(int rDirac, int colIndex, int rowIndex,const LLVectorType& LL)
{
	using real = typename LLVectorType::ValueType::Real;
	real d1; //dirac 1
	real d2; //dirac 2
	real d3; //dirac 3
	real ddd;

	d1 = diracDelta(rDirac,(LL[rowIndex].x - LL[colIndex].x));
	if (d1>0)
	{
		d2 = diracDelta(rDirac, (LL[rowIndex].y - LL[colIndex].y));
		if (d2>0)
		{
			d3=diracDelta(rDirac, (LL[rowIndex].z - LL[colIndex].z));
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
