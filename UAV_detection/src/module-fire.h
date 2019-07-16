//*************************************************************************
// Copyright (C) 2019 MOHA DALAB @ Illinois Institute of Technology

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************

#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>

#include "layer-conv2d.h"
#include "misc.h"

template <	unsigned pw_K,
			unsigned pw_S,
			unsigned pw_Din,
			unsigned pw_Cin,
			unsigned pw_Cout,
			unsigned pw_Ibit,
			unsigned pw_Wbit,
			unsigned pw_Mbit,
			unsigned pw_Abit,
			unsigned pw_MVTU_InP,
			unsigned pw_MVTU_OutP,
			
			unsigned conv1x1_K,
			unsigned conv1x1_S,
			unsigned conv1x1_Din,
			unsigned conv1x1_Cin,
			unsigned conv1x1_Cout,
			unsigned conv1x1_Ibit,
			unsigned conv1x1_Wbit,
			unsigned conv1x1_Mbit,
			unsigned conv1x1_Abit,
			unsigned conv1x1_MVTU_InP,
			unsigned conv1x1_MVTU_OutP,

			unsigned conv3x3_K,
			unsigned conv3x3_S,
			unsigned conv3x3_Din,
			unsigned conv3x3_Cin,
			unsigned conv3x3_Cout,
			unsigned conv3x3_Ibit,
			unsigned conv3x3_Wbit,
			unsigned conv3x3_Mbit,
			unsigned conv3x3_Abit,
			unsigned conv3x3_MVTU_InP,
			unsigned conv3x3_MVTU_OutP,

			unsigned ScaleBits>
void FIRE_ACT(
	stream<ap_uint<pw_Cin*pw_Ibit> >& in,

	const ap_uint<pw_MVTU_InP*pw_Wbit> pw_weights[pw_MVTU_OutP][((pw_Cin*pw_K*pw_K)/pw_MVTU_InP)*(pw_Cout/pw_MVTU_OutP)],
	const ap_int<pw_Mbit> pw_factorA[pw_MVTU_OutP][pw_Cout/pw_MVTU_OutP],
	const ap_int<pw_Mbit> pw_factorB[pw_MVTU_OutP][pw_Cout/pw_MVTU_OutP],

	const ap_uint<conv1x1_MVTU_InP*conv1x1_Wbit> conv1x1_weights[conv1x1_MVTU_OutP][((conv1x1_Cin*conv1x1_K*conv1x1_K)/conv1x1_MVTU_InP)*(conv1x1_Cout/conv1x1_MVTU_OutP)], 
	const ap_int<conv1x1_Mbit> conv1x1_factorA[conv1x1_MVTU_OutP][conv1x1_Cout/conv1x1_MVTU_OutP],
	const ap_int<conv1x1_Mbit> conv1x1_factorB[conv1x1_MVTU_OutP][conv1x1_Cout/conv1x1_MVTU_OutP],

	const ap_uint<conv3x3_MVTU_InP*conv3x3_Wbit> conv3x3_weights[conv3x3_MVTU_OutP][((conv3x3_Cin*conv3x3_K*conv3x3_K)/conv3x3_MVTU_InP)*(conv3x3_Cout/conv3x3_MVTU_OutP)], 
	const ap_int<conv3x3_Mbit> conv3x3_factorA[conv3x3_MVTU_OutP][conv3x3_Cout/conv3x3_MVTU_OutP],
	const ap_int<conv3x3_Mbit> conv3x3_factorB[conv3x3_MVTU_OutP][conv3x3_Cout/conv3x3_MVTU_OutP],

	stream<ap_uint<conv1x1_Cout*conv1x1_Abit+conv3x3_Cout*conv3x3_Abit> >& out, 
	const unsigned reps = 1)
{
	//static_assert( conv1x1_Din == conv3x3_Din, "For FIRE module, conv1x1_Din is not equal to conv3x3_Din");

#pragma HLS DATAFLOW

	stream<ap_uint<pw_Cout*pw_Abit> > pw_out("pw_out");
	CONV2D_ACT_NoP<pw_K, pw_S, pw_Din, pw_Cin, pw_Cout, pw_Ibit, pw_Wbit, pw_Mbit, pw_Abit, pw_MVTU_InP, pw_MVTU_OutP, ScaleBits>
	(in, pw_weights, pw_factorA, pw_factorB, pw_out, reps);
	
	stream<ap_uint<pw_Cout*pw_Abit> > pw_out1x1("pw_out1x1");
	stream<ap_uint<pw_Cout*pw_Abit> > pw_out3x3("pw_out3x3");
	DuplicateStreams<pw_Cout*pw_Abit, pw_Din*pw_Din>(pw_out, pw_out1x1, pw_out3x3, reps);

	stream<ap_uint<conv1x1_Cout*conv1x1_Abit> > conv_out1x1("conv_out1x1");
#pragma HLS STREAM variable=conv_out1x1 depth=conv3x3_Din*conv3x3_K+48
	CONV2D_ACT_NoP<conv1x1_K, conv1x1_S, conv1x1_Din, conv1x1_Cin, conv1x1_Cout, conv1x1_Ibit, conv1x1_Wbit, conv1x1_Mbit, conv1x1_Abit, conv1x1_MVTU_InP, conv1x1_MVTU_OutP, ScaleBits>
	(pw_out1x1, conv1x1_weights, conv1x1_factorA, conv1x1_factorB, conv_out1x1, reps);

	stream<ap_uint<conv3x3_Cout*conv3x3_Abit> > conv_out3x3("conv_out3x3");
	CONV2D_ACT_NoP<conv3x3_K, conv3x3_S, conv3x3_Din, conv3x3_Cin, conv3x3_Cout, conv3x3_Ibit, conv3x3_Wbit, conv3x3_Mbit, conv3x3_Abit, conv3x3_MVTU_InP, conv3x3_MVTU_OutP, ScaleBits>
	(pw_out3x3, conv3x3_weights, conv3x3_factorA, conv3x3_factorB, conv_out3x3, reps);

	ConcatStreams<conv1x1_Cout*conv1x1_Abit, conv3x3_Cout*conv3x3_Abit, conv1x1_Din*conv1x1_Din>(conv_out1x1, conv_out3x3, out, reps);
}

template <	unsigned pw_K,
			unsigned pw_S,
			unsigned pw_Din,
			unsigned pw_Cin,
			unsigned pw_Cout,
			unsigned pw_Ibit,
			unsigned pw_Wbit,
			unsigned pw_Mbit,
			unsigned pw_Abit,
			unsigned pw_MVTU_InP,
			unsigned pw_MVTU_OutP,
			
			unsigned conv1x1_K,
			unsigned conv1x1_S,
			unsigned conv1x1_Din,
			unsigned conv1x1_Cin,
			unsigned conv1x1_Cout,
			unsigned conv1x1_Ibit,
			unsigned conv1x1_Wbit,
			unsigned conv1x1_Mbit,
			unsigned conv1x1_MVTU_InP,
			unsigned conv1x1_MVTU_OutP,

			unsigned conv3x3_K,
			unsigned conv3x3_S,
			unsigned conv3x3_Din,
			unsigned conv3x3_Cin,
			unsigned conv3x3_Cout,
			unsigned conv3x3_Ibit,
			unsigned conv3x3_Wbit,
			unsigned conv3x3_Mbit,
			unsigned conv3x3_MVTU_InP,
			unsigned conv3x3_MVTU_OutP,

			unsigned ScaleBits>
void FIRE_NOACT(
	stream<ap_uint<pw_Cin*pw_Ibit> >& in,

	const ap_uint<pw_MVTU_InP*pw_Wbit> pw_weights[pw_MVTU_OutP][((pw_Cin*pw_K*pw_K)/pw_MVTU_InP)*(pw_Cout/pw_MVTU_OutP)],
	const ap_int<pw_Abit> pw_factorA[pw_MVTU_OutP][pw_Cout/pw_MVTU_OutP],
	const ap_int<pw_Abit> pw_factorB[pw_MVTU_OutP][pw_Cout/pw_MVTU_OutP],

	const ap_uint<conv1x1_MVTU_InP*conv1x1_Wbit> conv1x1_weights[conv1x1_MVTU_OutP][((conv1x1_Cin*conv1x1_K*conv1x1_K)/conv1x1_MVTU_InP)*(conv1x1_Cout/conv1x1_MVTU_OutP)], 

	const ap_uint<conv3x3_MVTU_InP*conv3x3_Wbit> conv3x3_weights[conv3x3_MVTU_OutP][((conv3x3_Cin*conv3x3_K*conv3x3_K)/conv3x3_MVTU_InP)*(conv3x3_Cout/conv3x3_MVTU_OutP)], 
	
	stream<ap_uint<conv1x1_Cout*conv1x1_Mbit+conv3x3_Cout*conv3x3_Mbit> >& out, 
	const unsigned reps = 1)
{
	//static_assert( conv1x1_Din == conv3x3_Din, "For FIRE module, conv1x1_Din is not equal to conv3x3_Din");

#pragma HLS DATAFLOW

	stream<ap_uint<pw_Cout*pw_Abit> > pw_out("pw_out");
	CONV2D_ACT_NoP<pw_K, pw_S, pw_Din, pw_Cin, pw_Cout, pw_Ibit, pw_Wbit, pw_Mbit, pw_Abit, pw_MVTU_InP, pw_MVTU_OutP, ScaleBits>
	(in, pw_weights, pw_factorA, pw_factorB, pw_out, reps);
	
	stream<ap_uint<pw_Cout*pw_Abit> > pw_out1x1("pw_out1x1");
	stream<ap_uint<pw_Cout*pw_Abit> > pw_out3x3("pw_out3x3");
	DuplicateStreams<pw_Cout*pw_Abit, pw_Din*pw_Din>(pw_out, pw_out1x1, pw_out3x3, reps);

	stream<ap_uint<conv1x1_Cout*conv1x1_Mbit> > conv_out1x1("conv_out1x1");
#pragma HLS STREAM variable=conv_out1x1 depth=conv3x3_Din*conv3x3_K+48
	CONV2D_NOACT_NoP<conv1x1_K, conv1x1_S, conv1x1_Din, conv1x1_Cin, conv1x1_Cout, conv1x1_Ibit, conv1x1_Wbit, conv1x1_Mbit, conv1x1_MVTU_InP, conv1x1_MVTU_OutP, ScaleBits>
	(pw_out1x1, conv1x1_weights, conv_out1x1, reps);

	stream<ap_uint<conv3x3_Cout*conv3x3_Mbit> > conv_out3x3("conv_out3x3");
	CONV2D_NOACT_NoP<conv3x3_K, conv3x3_S, conv3x3_Din, conv3x3_Cin, conv3x3_Cout, conv3x3_Ibit, conv3x3_Wbit, conv3x3_Mbit, conv3x3_MVTU_InP, conv3x3_MVTU_OutP, ScaleBits>
	(pw_out3x3, conv3x3_weights, conv_out3x3, reps);

	ConcatStreams<conv1x1_Cout*conv1x1_Mbit, conv3x3_Cout*conv3x3_Mbit, conv1x1_Din*conv1x1_Din>(conv_out1x1, conv_out3x3, out, reps);
}

template <	unsigned pw_K,
			unsigned pw_S,
			unsigned pw_MAX_Din,
			unsigned pw_MAX_Cin,
			unsigned pw_MAX_Cout,
			unsigned pw_Ibit,
			unsigned pw_Wbit,
			unsigned pw_Mbit,
			unsigned pw_Abit,
			unsigned pw_MVTU_InP,
			unsigned pw_MVTU_OutP,

			unsigned conv3x3_K,
			unsigned conv3x3_S,
			unsigned conv3x3_MAX_Din,
			unsigned conv3x3_MAX_Cin,
			unsigned conv3x3_MAX_Cout,
			unsigned conv3x3_Ibit,
			unsigned conv3x3_Wbit,
			unsigned conv3x3_Mbit,
			unsigned conv3x3_Abit,
			unsigned conv3x3_MVTU_InP,
			unsigned conv3x3_MVTU_OutP,

			unsigned ScaleBits,
			unsigned FactorScaleBits>
void HALFFIRE_ACT_variable(
	stream<ap_uint<pw_MAX_Cin*pw_Ibit> >& in,

	const ap_uint<pw_MVTU_InP*pw_Wbit> pw_weights[pw_MVTU_OutP][((pw_MAX_Cin*pw_K*pw_K)/pw_MVTU_InP)*(pw_MAX_Cout/pw_MVTU_OutP)],
	const ap_int<pw_Mbit> pw_factorA[pw_MVTU_OutP][pw_MAX_Cout/pw_MVTU_OutP],
	const ap_int<pw_Mbit> pw_factorB[pw_MVTU_OutP][pw_MAX_Cout/pw_MVTU_OutP],

	const ap_uint<conv3x3_MVTU_InP*conv3x3_Wbit> conv3x3_weights[conv3x3_MVTU_OutP][((conv3x3_MAX_Cin*conv3x3_K*conv3x3_K)/conv3x3_MVTU_InP)*(conv3x3_MAX_Cout/conv3x3_MVTU_OutP)], 
	const ap_int<conv3x3_Mbit> conv3x3_factorA[conv3x3_MVTU_OutP][conv3x3_MAX_Cout/conv3x3_MVTU_OutP],
	const ap_int<conv3x3_Mbit> conv3x3_factorB[conv3x3_MVTU_OutP][conv3x3_MAX_Cout/conv3x3_MVTU_OutP],

	stream<ap_uint<conv3x3_MAX_Cout*conv3x3_Abit> >& out,

	const unsigned pw_Din,
	// const unsigned pw_Cin,
	// const unsigned pw_Cout,

	const unsigned conv_Din,
	// const unsigned conv_Cin,
	// const unsigned conv_Cout,

	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
	stream<ap_uint<pw_MAX_Cout*pw_Abit> > pw_out("pw_out");
	CONV2D_1x1_ACT_NoP_variable<pw_MAX_Din, pw_MAX_Cin, pw_MAX_Cout, pw_Ibit, pw_Wbit, pw_Mbit, pw_Abit, pw_MVTU_InP, pw_MVTU_OutP, ScaleBits, FactorScaleBits>
	(in, pw_weights, pw_factorA, pw_factorB, pw_out, pw_Din, /*pw_Cin, pw_Cout,*/ reps);	

	CONV2D_ACT_NoP_variable<conv3x3_K, conv3x3_MAX_Din, conv3x3_MAX_Cin, conv3x3_MAX_Cout, conv3x3_Ibit, conv3x3_Wbit, conv3x3_Mbit, conv3x3_Abit, conv3x3_MVTU_InP, conv3x3_MVTU_OutP, ScaleBits, FactorScaleBits>
	(pw_out, conv3x3_weights, conv3x3_factorA, conv3x3_factorB, out, conv_Din, /*conv_Cin, conv_Cout,*/ reps);
}
