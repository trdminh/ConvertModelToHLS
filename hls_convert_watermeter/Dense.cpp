#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
#include <hls_math.h>
void Dense_0(fxp input_Dense[64],fxp &output_Dense_0,fxp bias[20],fxp weight[1280]){
	fxp out_Dense[20];
	loop_for_a_Dense_0:
	for (int i = 0; i < 20; i++){
		fxp s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 64; j++){
			s+=input_Dense[j]*weight[j*20+i];
		}
		out_Dense[i]=s+bias[i];
	}
	int maxindex = 0;
	fxp max=out_Dense[0];
	loop_detect:
	for (int i=0; i<20; i++){
		if (out_Dense[i]> max) {
			max=out_Dense[i];
			maxindex=i;
		}
	}
	fxp sum_exp_x = 0.0;
	for(int i = 0; i <20;i++){
		sum_exp_x += hls::exp(out_Dense[i]- out_Dense[maxindex]);
	}
	fxp max_value = out_Dense[maxindex];
	for(int i = 0; i <20;i++){
		out_Dense[i] = hls::exp(out_Dense[i] - max_value) / sum_exp_x;
	}
	fxp maxindex_2 = 0;
	fxp max_2 = out_Dense[0];
	for(int i = 0; i <20;i++){
		if (out_Dense[i] > max_2) {
			max_2 = out_Dense[i];
			maxindex_2 = i;
		}
	}
	output_Dense0 = maxindex_2;
}
