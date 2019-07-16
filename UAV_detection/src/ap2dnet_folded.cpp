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

#define TESTBENCH
#define REAL_IMAGE
// #define DEBUG

#define AP_INT_MAX_W 16384

#include "hls-nn-lib.h"
#include "ap2dnet-config.h"
#include "ap2dnet-params.h"
#include <stdint.h>
#include <stdlib.h>

#define L_K1 1
#define L_K3 3
#define L_S 1
#define L_MAX_Din 58
#define L_MAX_pw_Cin 160
#define L_MAX_pw_Cout 128
#define L_MAX_conv_Cin 128
#define L_MAX_conv_Cout 160
#define L_Ibit L1_Ibit
#define L_Wbit L1_Wbit
#define L_Mbit L1_Mbit
#define L_Abit L1_Abit
#define INPUT_WIDTH 512

#define pw_L_MVTU_InP L1_MVTU_InP
#define pw_L_MVTU_OutP L1_MVTU_OutP
#define conv_L_MVTU_InP L2_MVTU_InP
#define conv_L_MVTU_OutP L2_MVTU_OutP

#define USEFUL_LINE_BITS 480
#define LINES_PER_ALL_CHANNELS 1
const unsigned NumLinesPerRep = 3136;

#define LAST_LAYER 7

#define pw_WEIGHT_ITERATIONS ((L_MAX_pw_Cin*L_K1*L_K1)/pw_L_MVTU_InP)*(L_MAX_pw_Cout/pw_L_MVTU_OutP)
#define pw_FACTOR_ITERATIONS L_MAX_pw_Cout/pw_L_MVTU_OutP
#define conv_WEIGHT_ITERATIONS ((L_MAX_conv_Cin*L_K3*L_K3)/conv_L_MVTU_InP)*(L_MAX_conv_Cout/conv_L_MVTU_OutP)
#define conv_FACTOR_ITERATIONS L_MAX_conv_Cout/conv_L_MVTU_OutP
#define TOTAL_ITERATIONS pw_WEIGHT_ITERATIONS+pw_FACTOR_ITERATIONS+conv_WEIGHT_ITERATIONS+conv_FACTOR_ITERATIONS

static ap_uint<pw_L_MVTU_InP*L_Wbit> pw_weights[pw_L_MVTU_OutP][pw_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> pw_factorA[pw_L_MVTU_OutP][pw_FACTOR_ITERATIONS];
static ap_int<L_Mbit> pw_factorB[pw_L_MVTU_OutP][pw_FACTOR_ITERATIONS];

static ap_uint<conv_L_MVTU_InP*L_Wbit> conv3x3_weights[conv_L_MVTU_OutP][conv_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> conv3x3_factorA[conv_L_MVTU_OutP][conv_FACTOR_ITERATIONS]; 
static ap_int<L_Mbit> conv3x3_factorB[conv_L_MVTU_OutP][conv_FACTOR_ITERATIONS];

template <unsigned LineWidth, unsigned NumLines>
void DemuxStream3 (
	stream<ap_uint<LineWidth> >& in, 
	stream<ap_uint<LineWidth> >& out1, 
	stream<ap_uint<LineWidth> >& out2, 
	stream<ap_uint<LineWidth> >& out3, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp = in.read();
		if (whichFire == 1)
			out1.write(temp);
		else if (whichFire == 2)
			out2.write(temp);
		else
			out3.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void DemuxStream2 (
	stream<ap_uint<LineWidth> >& in, 
	stream<ap_uint<LineWidth> >& out1, 
	stream<ap_uint<LineWidth> >& out2, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp = in.read();
		if (whichFire == LAST_LAYER)
			out2.write(temp);
		else
			out1.write(temp);
	}
}

template <unsigned NumLines>
void DemuxStream2_0 (
	stream<ap_axis >& in, 
	stream<ap_axis >& out1, 
	stream<ap_axis >& out2, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_axis temp = in.read();
		if (whichFire == 1)
			out1.write(temp);
		else
			out2.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream3 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2, 
	stream<ap_uint<LineWidth> >& in3,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == 1)
			temp = in1.read();
		else if (whichFire == 2)
			temp = in2.read();
		else
			temp = in3.read();
		out.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == LAST_LAYER)
			temp = in2.read();
		else
			temp = in1.read();
		out.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2_0 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == 1)
			temp = in1.read();
		else
			temp = in2.read();
		out.write(temp);
	}
}

void DoFire(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned pw_Din, const unsigned pw_Cin, const unsigned pw_Cout,
	const unsigned conv_Din, const unsigned conv_Din_afterpool, const unsigned conv_Cin, const unsigned conv_Cout,
	const unsigned whichFire, const unsigned numReps,
	const unsigned first_numReps,
	const unsigned conv0_numReps,
	const unsigned other_numReps,
	const unsigned pool1_numReps,
	const unsigned pool2_numReps,
	const unsigned fire5_numReps,
	const unsigned main_out_numReps,
	const unsigned final_out_numReps) 
{
#pragma HLS DATAFLOW
	stream<ap_axis> to_conv0("to_conv0");
	stream<ap_axis> to_fire("to_fire");
	DemuxStream2_0<1>(in, to_conv0, to_fire, whichFire, first_numReps);

// BRANCH 1
	stream<ap_uint<384> > in_stream_extract0("DoCompute.in_stream_extract0");
	ExtractPixels<384, NumLinesPerRep> (to_conv0, in_stream_extract0, conv0_numReps);

	stream<ap_uint<L0_Cin*L0_Ibit> > in_stream("DoCompute.in_stream");
	ReduceWidth<384, L0_Cin*L0_Ibit, NumLinesPerRep> (in_stream_extract0, in_stream, conv0_numReps);
#ifdef DEBUG
	Monitor<L0_Din, L0_Cin, L0_Ibit>(in_stream, (char*)"./log/mon_in_stream_folded.log", conv0_numReps);
#endif
	stream<ap_uint<L0_Cout*L0_Abit> > conv1("conv1");
	CONV2D_ACT_NoP<L0_K, L0_S, L0_Din, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(in_stream, weights0, factorA0, factorB0, conv1, conv0_numReps);
#ifdef DEBUG
	if (whichFire == 1)
		Monitor<L0_Din/L0_S, L0_Cout, L0_Abit>(conv1, (char*)"log/mon_conv1_folded.log", conv0_numReps);
#endif
	stream<ap_uint<L18_Cin*L18_Ibit> > pool1("pool1");
	POOL2D_NoP<L18_K, L18_S, L18_Din, L18_Cin, L18_Ibit> (conv1, pool1, conv0_numReps);
	stream<ap_uint<L_MAX_pw_Cin*L_Ibit> > out_padded("out_padded");
	AppendZeros<L18_Cin*L18_Ibit, L_MAX_pw_Cin*L_Ibit, L1_Din*L1_Din> (pool1, out_padded, conv0_numReps);

// BRANCH 2
	stream<ap_uint<USEFUL_LINE_BITS> > in_stream_extract1("DoCompute.in_stream_extract1");
	ExtractPixels<USEFUL_LINE_BITS, LINES_PER_ALL_CHANNELS> (to_fire, in_stream_extract1, other_numReps);
	stream<ap_uint<L_MAX_pw_Cin*L_Ibit> > fire_in("fire_in");
	convWidth<USEFUL_LINE_BITS, L_MAX_pw_Cin*L_Ibit, 1> (in_stream_extract1, fire_in, other_numReps);


	stream<ap_uint<L_MAX_pw_Cin*L_Ibit> > first_out("first_out");
	MuxStream2_0<L_MAX_pw_Cin*L_Ibit, 1>(out_padded, fire_in, first_out, whichFire, pw_Din*pw_Din*numReps);


	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > fire_out("fire_out");
	HALFFIRE_ACT_variable<	L_K1, L_S, L_MAX_Din, L_MAX_pw_Cin, L_MAX_pw_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, pw_L_MVTU_InP, pw_L_MVTU_OutP,
							L_K3, L_S, L_MAX_Din, L_MAX_conv_Cin, L_MAX_conv_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, conv_L_MVTU_InP, conv_L_MVTU_OutP,
							SCALE_BITS, FACTOR_SCALE_BITS>
	(first_out, pw_weights, pw_factorA, pw_factorB, conv3x3_weights, conv3x3_factorA, conv3x3_factorB, fire_out, 
	pw_Din, /*pw_Cin, pw_Cout,*/ conv_Din, /*conv_Cin, conv_Cout,*/ numReps);

	

	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > fire_out1("fire_out1");
	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > fire_out2("fire_out2");
	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > fire_out3("fire_out3");
	DemuxStream3<L_MAX_conv_Cout*L_Abit, 1> (fire_out, fire_out1, fire_out2, fire_out3, whichFire, conv_Din*conv_Din*numReps);


	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > pool_out1("pool_out");
	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > pool_out2("pool_out");
	POOL2D_NoP<L19_K, L19_S, L19_Din, L_MAX_conv_Cout, L_Ibit> (fire_out1, pool_out1, pool1_numReps);
	POOL2D_NoP<L20_K, L20_S, L20_Din, L_MAX_conv_Cout, L_Ibit> (fire_out2, pool_out2, pool2_numReps);

	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > pool_out("pool_out");
	MuxStream3<L_MAX_conv_Cout*L_Abit, 1> (pool_out1, pool_out2, fire_out3, pool_out, whichFire, conv_Din_afterpool*conv_Din_afterpool*numReps);

#ifdef DEBUG
	if (whichFire == 1)
		Monitor<L19_Din/L19_S, L19_Cin, L19_Ibit>(pool_out, (char*)"log/mon_pool2_folded.log", numReps);
	else if (whichFire == 2)
		Monitor<L20_Din/L20_S, L20_Cin, L20_Ibit>(pool_out, (char*)"log/mon_pool3_folded.log", numReps);
	else if (whichFire == 5)
		Monitor<L10_Din/L10_S, L10_Cout, L10_Abit>(pool_out, (char*)"log/mon_fire5_folded.log", numReps);
	else if (whichFire == 6)
		Monitor<L12_Din/L12_S, L12_Cout, L12_Abit>(pool_out, (char*)"log/mon_fire6_folded.log", numReps);
	else if (whichFire == 7)
		Monitor<L14_Din/L14_S, L14_Cout, L14_Abit>(pool_out, (char*)"log/mon_fire7_folded.log", numReps);
#endif


	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > main_out("main_out");
	stream<ap_uint<L_MAX_conv_Cout*L_Abit> > fire5("fire5");
	DemuxStream2<L_MAX_conv_Cout*L_Abit, 1> (pool_out, main_out, fire5, whichFire, conv_Din_afterpool*conv_Din_afterpool*numReps);


// BRANCH 1
	stream<ap_uint<INPUT_WIDTH> > main_out_padded("main_out_padded");
	AppendZeros<USEFUL_LINE_BITS, INPUT_WIDTH, 1> (main_out, main_out_padded, LINES_PER_ALL_CHANNELS*conv_Din_afterpool*conv_Din_afterpool*main_out_numReps);

// BRANCH 2
	stream<ap_uint<L14_Cout*L14_Abit> > fire5_class("fire5_class");
	stream<ap_uint<L14_Cout*L14_Abit> > fire5_obj("fire5_obj");
#pragma HLS STREAM variable=fire5_obj depth=14*14+48
//#pragma HLS STREAM variable=fire5_obj depth=480
	DuplicateStreams<L14_Cout*L14_Abit, L15_Din*L15_Din>(fire5, fire5_class, fire5_obj, fire5_numReps);
	stream<ap_uint<L15_Cout*L15_Abit> > conv_class("conv_class");
	CONV2D_1x1_ACT_NoP<L15_Din, L15_Cin, L15_Cout, L15_Ibit, L15_Wbit, L15_Mbit, L15_Abit, L15_MVTU_InP, L15_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(fire5_class, weights15, factorA15, factorB15, conv_class, fire5_numReps);
#ifdef DEBUG
	Monitor<L15_Din/L15_S, L15_Cout, L15_Abit>(conv_class, (char*)"log/mon_conv_class_folded.log", fire5_numReps);
#endif
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out("class_out");
	ConcatStreams<L14_Cout*L14_Abit, L15_Cout*L15_Abit, L15_Din*L15_Din>(fire5_obj, conv_class, class_out, fire5_numReps);
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out_obj("class_out_obj");
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out_box("class_out_box");
	DuplicateStreams<(L14_Cout+L15_Cout)*L15_Abit, L16_Din*L16_Din>(class_out, class_out_obj, class_out_box, fire5_numReps);
	stream<ap_uint<L16_Cout*L16_Mbit> > conv_obj("conv_obj");
	CONV2D_1x1_NOACT_NoP<L16_Din, L16_Cin, L16_Cout, L16_Ibit, L16_Wbit, L16_Mbit, L16_MVTU_InP, L16_MVTU_OutP>
	(class_out_obj, weights16, conv_obj, fire5_numReps);
	stream<ap_uint<L17_Cout*L17_Mbit> > conv_box("conv_box");
#pragma HLS STREAM variable=conv_box depth=14*14
//#pragma HLS STREAM variable=conv_box depth=480
	CONV2D_1x1_NOACT_NoP<L17_Din, L17_Cin, L17_Cout, L17_Ibit, L17_Wbit, L17_Mbit, L17_MVTU_InP, L17_MVTU_OutP>
	(class_out_box, weights17, conv_box, fire5_numReps);
	stream<ap_uint<8+L17_Cout*L17_Mbit> > box_prediction("box_prediction");
	ObjDetectSelect<L17_Mbit, L17_Cout*L17_Mbit, L17_Din*L17_Din> (conv_obj, conv_box, box_prediction, fire5_numReps);
	stream<ap_uint<INPUT_WIDTH> > box_prediction_padded("box_prediction_padded");
	AppendZeros<8+L17_Cout*L17_Mbit, INPUT_WIDTH, 1> (box_prediction, box_prediction_padded, fire5_numReps);

	stream<ap_uint<INPUT_WIDTH> > final_out("final_out");
	MuxStream2<INPUT_WIDTH, 1>(main_out_padded, box_prediction_padded, final_out, whichFire, final_out_numReps);

	AddLast<1> (final_out, out, final_out_numReps);
}

void writeWeightsFactors(stream<ap_axis >& in) {
#pragma HLS DATAFLOW

	stream<ap_uint<INPUT_WIDTH> > pw_weights_stream("pw_weights_stream");
	stream<ap_uint<INPUT_WIDTH> > pw_factors_stream("pw_weights_stream");
	stream<ap_uint<INPUT_WIDTH> > conv_weights_stream("pw_weights_stream");
	stream<ap_uint<INPUT_WIDTH> > conv_factors_stream("pw_weights_stream");

	for (unsigned i = 0; i < TOTAL_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_axis temp_in = in.read();
		if (i < pw_WEIGHT_ITERATIONS)
			pw_weights_stream.write(temp_in.data);
		else if (i < pw_WEIGHT_ITERATIONS + conv_WEIGHT_ITERATIONS)
			conv_weights_stream.write(temp_in.data);
		else if (i < pw_WEIGHT_ITERATIONS + conv_WEIGHT_ITERATIONS + pw_FACTOR_ITERATIONS)
			pw_factors_stream.write(temp_in.data);
		else
			conv_factors_stream.write(temp_in.data);
	}

	for (unsigned i = 0; i < pw_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<INPUT_WIDTH> temp_in = pw_weights_stream.read();
		for (unsigned p = 0; p < pw_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<pw_L_MVTU_InP*L_Wbit> temp = temp_in( (p+1)*pw_L_MVTU_InP*L_Wbit-1, p*pw_L_MVTU_InP*L_Wbit );
			pw_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < conv_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<INPUT_WIDTH> temp_in = conv_weights_stream.read();
		for (unsigned p = 0; p < conv_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<conv_L_MVTU_InP*L_Wbit> temp = temp_in( (p+1)*conv_L_MVTU_InP*L_Wbit-1, p*conv_L_MVTU_InP*L_Wbit );
			conv3x3_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < pw_FACTOR_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<INPUT_WIDTH> temp_in = pw_factors_stream.read();
		for (unsigned p = 0; p < pw_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<2*L_Mbit> temp_factorAB = temp_in( (p+1)*2*L_Mbit-1, p*2*L_Mbit );
			pw_factorA[p][i] = temp_factorAB(L_Mbit-1, 0);
			pw_factorB[p][i] = temp_factorAB(2*L_Mbit-1, L_Mbit);
		}
	}

	for (unsigned i = 0; i < conv_FACTOR_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<INPUT_WIDTH> temp_in = conv_factors_stream.read();
		for (unsigned p = 0; p < conv_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<2*L_Mbit> temp_factorAB = temp_in( (p+1)*2*L_Mbit-1, p*2*L_Mbit );
			conv3x3_factorA[p][i] = temp_factorAB(L_Mbit-1, 0);
			conv3x3_factorB[p][i] = temp_factorAB(2*L_Mbit-1, L_Mbit);
		}
	}
}

void halfpwnet(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned pw_Din, const unsigned pw_Cin, const unsigned pw_Cout,
	const unsigned pw_weight_iterations, const unsigned pw_factor_iterations,
	const unsigned conv_Din, const unsigned conv_Din_afterpool, const unsigned conv_Cin, const unsigned conv_Cout,
	const unsigned conv_weight_iterations, const unsigned conv_factor_iterations,
	const unsigned whichFire, const unsigned numReps) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=pw_Din bundle=control
#pragma HLS INTERFACE s_axilite port=pw_Cin bundle=control
#pragma HLS INTERFACE s_axilite port=pw_Cout bundle=control
#pragma HLS INTERFACE s_axilite port=pw_weight_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=pw_factor_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=conv_Din bundle=control
#pragma HLS INTERFACE s_axilite port=conv_Din_afterpool bundle=control
#pragma HLS INTERFACE s_axilite port=conv_Cin bundle=control
#pragma HLS INTERFACE s_axilite port=conv_Cout bundle=control
#pragma HLS INTERFACE s_axilite port=conv_weight_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=conv_factor_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=whichFire bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0
#pragma HLS RESOURCE variable=weights15 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA15 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB15 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights15 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA15 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB15 complete dim=0
#pragma HLS RESOURCE variable=weights16 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA16 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB16 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights16 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA16 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB16 complete dim=0
#pragma HLS RESOURCE variable=weights17 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA17 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB17 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights17 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA17 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB17 complete dim=0
	
#pragma HLS RESOURCE variable=pw_weights core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=pw_factorA core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=pw_factorB core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=pw_weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=pw_factorA complete dim=0
#pragma HLS ARRAY_PARTITION variable=pw_factorB complete dim=0

#pragma HLS RESOURCE variable=conv3x3_weights core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=conv3x3_factorA core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=conv3x3_factorB core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=conv3x3_weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=conv3x3_factorA complete dim=0
#pragma HLS ARRAY_PARTITION variable=conv3x3_factorB complete dim=0

	const unsigned first_numReps = (whichFire == 1) ? NumLinesPerRep*numReps : LINES_PER_ALL_CHANNELS*pw_Din*pw_Din*numReps;
	const unsigned conv0_numReps = (whichFire == 1) ? numReps : 0;
	const unsigned other_numReps = (whichFire != 1) ? pw_Din*pw_Din*numReps : 0;
	const unsigned pool1_numReps = (whichFire == 1) ? numReps : 0;
	const unsigned pool2_numReps = (whichFire == 2) ? numReps : 0;
	const unsigned fire5_numReps = (whichFire == LAST_LAYER) ? numReps : 0;
	const unsigned main_out_numReps = (whichFire != LAST_LAYER) ? numReps : 0;
	const unsigned final_out_numReps = (whichFire == LAST_LAYER) ? numReps : LINES_PER_ALL_CHANNELS*conv_Din_afterpool*conv_Din_afterpool*numReps;

	if (whichFire == 13) {
		writeWeightsFactors(in);
	}
	else {
		DoFire(in, out,
			pw_Din, pw_Cin, pw_Cout,
			conv_Din, conv_Din_afterpool, conv_Cin, conv_Cout,
			whichFire, numReps,
			first_numReps,
			conv0_numReps,
			other_numReps,
			pool1_numReps,
			pool2_numReps,
			fire5_numReps,
			main_out_numReps,
			final_out_numReps);
	}
}

void II_determiner(stream<ap_axis >& in, stream<ap_axis >& out) {
	ap2dnet(in, out, L1_Din, L1_Cin, L1_Cout, 0, 0, L2_Din, L2_Din >> 1, L2_Cin, L2_Cout, 0, 0, 1, 1);
}


