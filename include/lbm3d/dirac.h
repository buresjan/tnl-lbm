#pragma once

#include "defs.h"

template< typename real >
CUDA_HOSTDEV bool isDDNonZero(int i, real r)
{
	switch (i)
	{
		case 1: // VU: phi3
			if (fabs(r) < (real)1.0)
				return true;
			else
				return false;
		case 2: // VU: phi2
			if (fabs(r) < (real)2.0)
				return true;
			else
				return false;
		case 3: // VU: phi1
			if (fabs(r) >= (real)2.0)
				return false;
			else
				return true;
		case 4: // VU: phi4
			if (fabs(r) >= (real)1.5)
				return false;
			else
				return true;
	}
	return false;
}


//TODO: Remove unnecessary isDDNonZero checks
template< typename real >
CUDA_HOSTDEV real diracDelta(int i, real r)
{
	if(!isDDNonZero(i, r))
	{
		return 0;
	}
	else{
		switch (i)
		{
			case 1: // VU: phi3
				return 1 - fabs(r);
			case 2: // VU: phi2
				return (real)0.25 * (1 + cos((real)PI*r*(real)0.5));
			case 3: // VU: phi1
				if (fabs(r) > (real)1.0)
					return (5 - 2*fabs(r) - sqrt(-7 + 12*fabs(r) - 4*r*r)) / (real)8.0;
				else
					return (3 - 2*fabs(r) + sqrt(1 + 4*fabs(r) - 4*r*r)) / (real)8.0;
			case 4: // VU: phi4
				if (fabs(r) > (real)0.5)
					return (5 - 3*fabs(r) - sqrt(-2 + 6*fabs(r) - 3*r*r)) / (real)6.0;
				else
					return (1 + sqrt(1 - 3*r*r)) / (real)3.0;
		}
	}

	// Just a failsafe in case something does not return 0
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

	d1 = diracDelta(rDirac,(LL[rowIndex].x - LL[colIndex].x));
	if (d1>0)
	{
		d2 = diracDelta(rDirac, (LL[rowIndex].y - LL[colIndex].y));
		if (d2>0)
		{
			d3=diracDelta(rDirac, (LL[rowIndex].z - LL[colIndex].z));
			if (d3>0)
			{
				return d1*d2*d3;
			}
		}
	}
	return 0;
}
