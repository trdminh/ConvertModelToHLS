#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Padding_Conv2D_0(fxp input_Pad_Conv[1024], fxp output_Pad_Conv[1156]){
	loop_for_3_channel_pad_0:
	for (int c = 0; c < 1; c++){
		loop_for_channel_pad_0:
		for (int n = 0; n < 34; n++){
			loop_for_weight_pad_0:
			for (int i = 0; i < 34; i++){
				if (n < 1 || n >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0;
				 else 
					if (i < 1 || i >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0; else output_Pad_Conv[34 * 34 * c + 34 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * (n - 1) + i - 1];
			}
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Conv2D_0(fxp Input_Conv[1156],fxp Output_Conv[16384], fxp kernel[144]){
	loop_for_channel2D_0:
	int stride = 1;
	for (int n = 0; n < 16; n++){
		loop_for_bp2D_0:
		for (int x = 0; x < 32; x++){
			loop_for_ap2D_0:
			for (int y = 0; y < 32; y++){
				fxp s = 0;
				loop_for_fc_0:
				for (int k = 0; k < 1; k++){
					loop_for_fb_0:
					for (int i = 0; i < 3; i++){
						loop_for_fa_0:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[1*3*3*n+3*3*k+3*i+j])*(Input_Conv[34*34*k+34*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[32*32*n+32*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_0(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 16; i++){
		for(int j = 0; j < 1024; j++){
			 Output_BatchNorm[1024 * i + j] = ((Input_BatchNorm[1024 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation0(fxp Input_Activation[16384], fxp Output_Activation[16384]){
	for (int i = 0; i < 16384; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_1(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]){
	loop_for_3_channel_pad_1:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_1:
		for (int n = 0; n < 34; n++){
			loop_for_weight_pad_1:
			for (int i = 0; i < 34; i++){
				if (n < 1 || n >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0;
				 else 
					if (i < 1 || i >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0; else output_Pad_Conv[34 * 34 * c + 34 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_1(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]){
	loop_for_channel2D_1:
	int stride = 1;
	for (int n = 0; n < 16; n++){
		loop_for_bp2D_1:
		for (int x = 0; x < 32; x++){
			loop_for_ap2D_1:
			for (int y = 0; y < 32; y++){
				fxp s = 0;
				loop_for_fc_1:
				for (int k = 0; k < 16; k++){
					loop_for_fb_1:
					for (int i = 0; i < 3; i++){
						loop_for_fa_1:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[16*3*3*n+3*3*k+3*i+j])*(Input_Conv[34*34*k+34*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[32*32*n+32*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_1(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 16; i++){
		for(int j = 0; j < 1024; j++){
			 Output_BatchNorm[1024 * i + j] = ((Input_BatchNorm[1024 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation1(fxp Input_Activation[16384], fxp Output_Activation[16384]){
	for (int i = 0; i < 16384; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_2(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]){
	loop_for_3_channel_pad_2:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_2:
		for (int n = 0; n < 34; n++){
			loop_for_weight_pad_2:
			for (int i = 0; i < 34; i++){
				if (n < 1 || n >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0;
				 else 
					if (i < 1 || i >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0; else output_Pad_Conv[34 * 34 * c + 34 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_2(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]){
	loop_for_channel2D_2:
	int stride = 1;
	for (int n = 0; n < 16; n++){
		loop_for_bp2D_2:
		for (int x = 0; x < 32; x++){
			loop_for_ap2D_2:
			for (int y = 0; y < 32; y++){
				fxp s = 0;
				loop_for_fc_2:
				for (int k = 0; k < 16; k++){
					loop_for_fb_2:
					for (int i = 0; i < 3; i++){
						loop_for_fa_2:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[16*3*3*n+3*3*k+3*i+j])*(Input_Conv[34*34*k+34*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[32*32*n+32*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_2(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 16; i++){
		for(int j = 0; j < 1024; j++){
			 Output_BatchNorm[1024 * i + j] = ((Input_BatchNorm[1024 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_0(fxp input_0[16384], fxp input_1[16384], fxp output[16384]) {
	for (int i = 0; i < 16384; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation2(fxp Input_Activation[16384], fxp Output_Activation[16384]){
	for (int i = 0; i < 16384; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_3(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]){
	loop_for_3_channel_pad_3:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_3:
		for (int n = 0; n < 34; n++){
			loop_for_weight_pad_3:
			for (int i = 0; i < 34; i++){
				if (n < 1 || n >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0;
				 else 
					if (i < 1 || i >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0; else output_Pad_Conv[34 * 34 * c + 34 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_3(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]){
	loop_for_channel2D_3:
	int stride = 1;
	for (int n = 0; n < 16; n++){
		loop_for_bp2D_3:
		for (int x = 0; x < 32; x++){
			loop_for_ap2D_3:
			for (int y = 0; y < 32; y++){
				fxp s = 0;
				loop_for_fc_3:
				for (int k = 0; k < 16; k++){
					loop_for_fb_3:
					for (int i = 0; i < 3; i++){
						loop_for_fa_3:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[16*3*3*n+3*3*k+3*i+j])*(Input_Conv[34*34*k+34*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[32*32*n+32*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_3(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 16; i++){
		for(int j = 0; j < 1024; j++){
			 Output_BatchNorm[1024 * i + j] = ((Input_BatchNorm[1024 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation3(fxp Input_Activation[16384], fxp Output_Activation[16384]){
	for (int i = 0; i < 16384; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_4(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]){
	loop_for_3_channel_pad_4:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_4:
		for (int n = 0; n < 34; n++){
			loop_for_weight_pad_4:
			for (int i = 0; i < 34; i++){
				if (n < 1 || n >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0;
				 else 
					if (i < 1 || i >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0; else output_Pad_Conv[34 * 34 * c + 34 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_4(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]){
	loop_for_channel2D_4:
	int stride = 1;
	for (int n = 0; n < 16; n++){
		loop_for_bp2D_4:
		for (int x = 0; x < 32; x++){
			loop_for_ap2D_4:
			for (int y = 0; y < 32; y++){
				fxp s = 0;
				loop_for_fc_4:
				for (int k = 0; k < 16; k++){
					loop_for_fb_4:
					for (int i = 0; i < 3; i++){
						loop_for_fa_4:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[16*3*3*n+3*3*k+3*i+j])*(Input_Conv[34*34*k+34*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[32*32*n+32*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_4(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 16; i++){
		for(int j = 0; j < 1024; j++){
			 Output_BatchNorm[1024 * i + j] = ((Input_BatchNorm[1024 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_1(fxp input_0[16384], fxp input_1[16384], fxp output[16384]) {
	for (int i = 0; i < 16384; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation4(fxp Input_Activation[16384], fxp Output_Activation[16384]){
	for (int i = 0; i < 16384; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_5(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]){
	loop_for_3_channel_pad_5:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_5:
		for (int n = 0; n < 34; n++){
			loop_for_weight_pad_5:
			for (int i = 0; i < 34; i++){
				if (n < 1 || n >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0;
				 else 
					if (i < 1 || i >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0; else output_Pad_Conv[34 * 34 * c + 34 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_5(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]){
	loop_for_channel2D_5:
	int stride = 1;
	for (int n = 0; n < 16; n++){
		loop_for_bp2D_5:
		for (int x = 0; x < 32; x++){
			loop_for_ap2D_5:
			for (int y = 0; y < 32; y++){
				fxp s = 0;
				loop_for_fc_5:
				for (int k = 0; k < 16; k++){
					loop_for_fb_5:
					for (int i = 0; i < 3; i++){
						loop_for_fa_5:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[16*3*3*n+3*3*k+3*i+j])*(Input_Conv[34*34*k+34*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[32*32*n+32*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_5(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 16; i++){
		for(int j = 0; j < 1024; j++){
			 Output_BatchNorm[1024 * i + j] = ((Input_BatchNorm[1024 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation5(fxp Input_Activation[16384], fxp Output_Activation[16384]){
	for (int i = 0; i < 16384; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_6(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]){
	loop_for_3_channel_pad_6:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_6:
		for (int n = 0; n < 34; n++){
			loop_for_weight_pad_6:
			for (int i = 0; i < 34; i++){
				if (n < 1 || n >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0;
				 else 
					if (i < 1 || i >= 33) output_Pad_Conv[34 * 34 * c + 34 * n + i]=0; else output_Pad_Conv[34 * 34 * c + 34 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_6(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]){
	loop_for_channel2D_6:
	int stride = 1;
	for (int n = 0; n < 16; n++){
		loop_for_bp2D_6:
		for (int x = 0; x < 32; x++){
			loop_for_ap2D_6:
			for (int y = 0; y < 32; y++){
				fxp s = 0;
				loop_for_fc_6:
				for (int k = 0; k < 16; k++){
					loop_for_fb_6:
					for (int i = 0; i < 3; i++){
						loop_for_fa_6:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[16*3*3*n+3*3*k+3*i+j])*(Input_Conv[34*34*k+34*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[32*32*n+32*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_6(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 16; i++){
		for(int j = 0; j < 1024; j++){
			 Output_BatchNorm[1024 * i + j] = ((Input_BatchNorm[1024 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_2(fxp input_0[16384], fxp input_1[16384], fxp output[16384]) {
	for (int i = 0; i < 16384; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation6(fxp Input_Activation[16384], fxp Output_Activation[16384]){
	for (int i = 0; i < 16384; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_7(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[17424]){
	loop_for_3_channel_pad_7:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_7:
		for (int n = 0; n < 33; n++){
			loop_for_weight_pad_7:
			for (int i = 0; i < 33; i++){
				if (n >= 32) output_Pad_Conv[33 * 33 * c + 33 * n + i]=0;
				 else 
					if (i >= 32) output_Pad_Conv[33 * 33 * c + 33 * n + i]=0; else output_Pad_Conv[33 * 33 * c + 33 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * n + i];
			}
		}
	}
}
void Conv2D_7(fxp Input_Conv[17424],fxp Output_Conv[8192], fxp kernel[4608]){
	loop_for_channel2D_7:
	int stride = 2;
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_7:
		for (int x = 0; x < 16; x++){
			loop_for_ap2D_7:
			for (int y = 0; y < 16; y++){
				fxp s = 0;
				loop_for_fc_7:
				for (int k = 0; k < 16; k++){
					loop_for_fb_7:
					for (int i = 0; i < 3; i++){
						loop_for_fa_7:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[16*3*3*n+3*3*k+3*i+j])*(Input_Conv[33*33*k+33*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[16*16*n+16*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_7(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			 Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation7(fxp Input_Activation[8192], fxp Output_Activation[8192]){
	for (int i = 0; i < 8192; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_8(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]){
	loop_for_3_channel_pad_8:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_8:
		for (int n = 0; n < 18; n++){
			loop_for_weight_pad_8:
			for (int i = 0; i < 18; i++){
				if (n < 1 || n >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0;
				 else 
					if (i < 1 || i >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0; else output_Pad_Conv[18 * 18 * c + 18 * n + i] = input_Pad_Conv[16 * 16 * c + 16 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_8(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]){
	loop_for_channel2D_8:
	int stride = 1;
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_8:
		for (int x = 0; x < 16; x++){
			loop_for_ap2D_8:
			for (int y = 0; y < 16; y++){
				fxp s = 0;
				loop_for_fc_8:
				for (int k = 0; k < 32; k++){
					loop_for_fb_8:
					for (int i = 0; i < 3; i++){
						loop_for_fa_8:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[18*18*k+18*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[16*16*n+16*x+y]=s;
			}
		}
	}
}
void Padding_Conv2D_9(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[15376]){
	loop_for_3_channel_pad_9:
	for (int c = 0; c < 16; c++){
		loop_for_channel_pad_9:
		for (int n = 0; n < 31; n++){
			loop_for_weight_pad_9:
			for (int i = 0; i < 31; i++){
				if (n >= 32) output_Pad_Conv[31 * 31 * c + 31 * n + i]=0;
				 else 
					if (i >= 32) output_Pad_Conv[31 * 31 * c + 31 * n + i]=0; else output_Pad_Conv[31 * 31 * c + 31 * n + i] = input_Pad_Conv[32 * 32 * c + 32 * n + i];
			}
		}
	}
}
void Conv2D_9(fxp Input_Conv[15376],fxp Output_Conv[8192], fxp kernel[512]){
	loop_for_channel2D_9:
	int stride = 2;
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_9:
		for (int x = 0; x < 16; x++){
			loop_for_ap2D_9:
			for (int y = 0; y < 16; y++){
				fxp s = 0;
				loop_for_fc_9:
				for (int k = 0; k < 16; k++){
					loop_for_fb_9:
					for (int i = 0; i < 1; i++){
						loop_for_fa_9:
						for (int j = 0; j < 1; j++){
							s=s+(kernel[16*1*1*n+1*1*k+1*i+j])*(Input_Conv[31*31*k+31*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[16*16*n+16*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_8(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			 Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_9(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			 Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_3(fxp input_0[8192], fxp input_1[8192], fxp output[8192]) {
	for (int i = 0; i < 8192; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation8(fxp Input_Activation[8192], fxp Output_Activation[8192]){
	for (int i = 0; i < 8192; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_10(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]){
	loop_for_3_channel_pad_10:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_10:
		for (int n = 0; n < 18; n++){
			loop_for_weight_pad_10:
			for (int i = 0; i < 18; i++){
				if (n < 1 || n >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0;
				 else 
					if (i < 1 || i >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0; else output_Pad_Conv[18 * 18 * c + 18 * n + i] = input_Pad_Conv[16 * 16 * c + 16 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_10(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]){
	loop_for_channel2D_10:
	int stride = 1;
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_10:
		for (int x = 0; x < 16; x++){
			loop_for_ap2D_10:
			for (int y = 0; y < 16; y++){
				fxp s = 0;
				loop_for_fc_10:
				for (int k = 0; k < 32; k++){
					loop_for_fb_10:
					for (int i = 0; i < 3; i++){
						loop_for_fa_10:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[18*18*k+18*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[16*16*n+16*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_10(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			 Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation9(fxp Input_Activation[8192], fxp Output_Activation[8192]){
	for (int i = 0; i < 8192; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_11(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]){
	loop_for_3_channel_pad_11:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_11:
		for (int n = 0; n < 18; n++){
			loop_for_weight_pad_11:
			for (int i = 0; i < 18; i++){
				if (n < 1 || n >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0;
				 else 
					if (i < 1 || i >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0; else output_Pad_Conv[18 * 18 * c + 18 * n + i] = input_Pad_Conv[16 * 16 * c + 16 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_11(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]){
	loop_for_channel2D_11:
	int stride = 1;
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_11:
		for (int x = 0; x < 16; x++){
			loop_for_ap2D_11:
			for (int y = 0; y < 16; y++){
				fxp s = 0;
				loop_for_fc_11:
				for (int k = 0; k < 32; k++){
					loop_for_fb_11:
					for (int i = 0; i < 3; i++){
						loop_for_fa_11:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[18*18*k+18*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[16*16*n+16*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_11(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			 Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_4(fxp input_0[8192], fxp input_1[8192], fxp output[8192]) {
	for (int i = 0; i < 8192; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation10(fxp Input_Activation[8192], fxp Output_Activation[8192]){
	for (int i = 0; i < 8192; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_12(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]){
	loop_for_3_channel_pad_12:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_12:
		for (int n = 0; n < 18; n++){
			loop_for_weight_pad_12:
			for (int i = 0; i < 18; i++){
				if (n < 1 || n >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0;
				 else 
					if (i < 1 || i >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0; else output_Pad_Conv[18 * 18 * c + 18 * n + i] = input_Pad_Conv[16 * 16 * c + 16 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_12(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]){
	loop_for_channel2D_12:
	int stride = 1;
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_12:
		for (int x = 0; x < 16; x++){
			loop_for_ap2D_12:
			for (int y = 0; y < 16; y++){
				fxp s = 0;
				loop_for_fc_12:
				for (int k = 0; k < 32; k++){
					loop_for_fb_12:
					for (int i = 0; i < 3; i++){
						loop_for_fa_12:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[18*18*k+18*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[16*16*n+16*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_12(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			 Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation11(fxp Input_Activation[8192], fxp Output_Activation[8192]){
	for (int i = 0; i < 8192; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_13(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]){
	loop_for_3_channel_pad_13:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_13:
		for (int n = 0; n < 18; n++){
			loop_for_weight_pad_13:
			for (int i = 0; i < 18; i++){
				if (n < 1 || n >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0;
				 else 
					if (i < 1 || i >= 17) output_Pad_Conv[18 * 18 * c + 18 * n + i]=0; else output_Pad_Conv[18 * 18 * c + 18 * n + i] = input_Pad_Conv[16 * 16 * c + 16 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_13(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]){
	loop_for_channel2D_13:
	int stride = 1;
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_13:
		for (int x = 0; x < 16; x++){
			loop_for_ap2D_13:
			for (int y = 0; y < 16; y++){
				fxp s = 0;
				loop_for_fc_13:
				for (int k = 0; k < 32; k++){
					loop_for_fb_13:
					for (int i = 0; i < 3; i++){
						loop_for_fa_13:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[18*18*k+18*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[16*16*n+16*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_13(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			 Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_5(fxp input_0[8192], fxp input_1[8192], fxp output[8192]) {
	for (int i = 0; i < 8192; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation12(fxp Input_Activation[8192], fxp Output_Activation[8192]){
	for (int i = 0; i < 8192; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_14(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[9248]){
	loop_for_3_channel_pad_14:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_14:
		for (int n = 0; n < 17; n++){
			loop_for_weight_pad_14:
			for (int i = 0; i < 17; i++){
				if (n >= 16) output_Pad_Conv[17 * 17 * c + 17 * n + i]=0;
				 else 
					if (i >= 16) output_Pad_Conv[17 * 17 * c + 17 * n + i]=0; else output_Pad_Conv[17 * 17 * c + 17 * n + i] = input_Pad_Conv[16 * 16 * c + 16 * n + i];
			}
		}
	}
}
void Conv2D_14(fxp Input_Conv[9248],fxp Output_Conv[4096], fxp kernel[18432]){
	loop_for_channel2D_14:
	int stride = 2;
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_14:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_14:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_14:
				for (int k = 0; k < 32; k++){
					loop_for_fb_14:
					for (int i = 0; i < 3; i++){
						loop_for_fa_14:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[17*17*k+17*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[8*8*n+8*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_14(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 64; j++){
			 Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation13(fxp Input_Activation[4096], fxp Output_Activation[4096]){
	for (int i = 0; i < 4096; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_15(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]){
	loop_for_3_channel_pad_15:
	for (int c = 0; c < 64; c++){
		loop_for_channel_pad_15:
		for (int n = 0; n < 10; n++){
			loop_for_weight_pad_15:
			for (int i = 0; i < 10; i++){
				if (n < 1 || n >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0;
				 else 
					if (i < 1 || i >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0; else output_Pad_Conv[10 * 10 * c + 10 * n + i] = input_Pad_Conv[8 * 8 * c + 8 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_15(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]){
	loop_for_channel2D_15:
	int stride = 1;
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_15:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_15:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_15:
				for (int k = 0; k < 64; k++){
					loop_for_fb_15:
					for (int i = 0; i < 3; i++){
						loop_for_fa_15:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[64*3*3*n+3*3*k+3*i+j])*(Input_Conv[10*10*k+10*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[8*8*n+8*x+y]=s;
			}
		}
	}
}
void Padding_Conv2D_16(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[7200]){
	loop_for_3_channel_pad_16:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_16:
		for (int n = 0; n < 15; n++){
			loop_for_weight_pad_16:
			for (int i = 0; i < 15; i++){
				if (n >= 16) output_Pad_Conv[15 * 15 * c + 15 * n + i]=0;
				 else 
					if (i >= 16) output_Pad_Conv[15 * 15 * c + 15 * n + i]=0; else output_Pad_Conv[15 * 15 * c + 15 * n + i] = input_Pad_Conv[16 * 16 * c + 16 * n + i];
			}
		}
	}
}
void Conv2D_16(fxp Input_Conv[7200],fxp Output_Conv[4096], fxp kernel[2048]){
	loop_for_channel2D_16:
	int stride = 2;
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_16:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_16:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_16:
				for (int k = 0; k < 32; k++){
					loop_for_fb_16:
					for (int i = 0; i < 1; i++){
						loop_for_fa_16:
						for (int j = 0; j < 1; j++){
							s=s+(kernel[32*1*1*n+1*1*k+1*i+j])*(Input_Conv[15*15*k+15*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[8*8*n+8*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_15(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 64; j++){
			 Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_16(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 64; j++){
			 Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_6(fxp input_0[4096], fxp input_1[4096], fxp output[4096]) {
	for (int i = 0; i < 4096; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation14(fxp Input_Activation[4096], fxp Output_Activation[4096]){
	for (int i = 0; i < 4096; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_17(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]){
	loop_for_3_channel_pad_17:
	for (int c = 0; c < 64; c++){
		loop_for_channel_pad_17:
		for (int n = 0; n < 10; n++){
			loop_for_weight_pad_17:
			for (int i = 0; i < 10; i++){
				if (n < 1 || n >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0;
				 else 
					if (i < 1 || i >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0; else output_Pad_Conv[10 * 10 * c + 10 * n + i] = input_Pad_Conv[8 * 8 * c + 8 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_17(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]){
	loop_for_channel2D_17:
	int stride = 1;
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_17:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_17:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_17:
				for (int k = 0; k < 64; k++){
					loop_for_fb_17:
					for (int i = 0; i < 3; i++){
						loop_for_fa_17:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[64*3*3*n+3*3*k+3*i+j])*(Input_Conv[10*10*k+10*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[8*8*n+8*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_17(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 64; j++){
			 Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation15(fxp Input_Activation[4096], fxp Output_Activation[4096]){
	for (int i = 0; i < 4096; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_18(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]){
	loop_for_3_channel_pad_18:
	for (int c = 0; c < 64; c++){
		loop_for_channel_pad_18:
		for (int n = 0; n < 10; n++){
			loop_for_weight_pad_18:
			for (int i = 0; i < 10; i++){
				if (n < 1 || n >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0;
				 else 
					if (i < 1 || i >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0; else output_Pad_Conv[10 * 10 * c + 10 * n + i] = input_Pad_Conv[8 * 8 * c + 8 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_18(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]){
	loop_for_channel2D_18:
	int stride = 1;
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_18:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_18:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_18:
				for (int k = 0; k < 64; k++){
					loop_for_fb_18:
					for (int i = 0; i < 3; i++){
						loop_for_fa_18:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[64*3*3*n+3*3*k+3*i+j])*(Input_Conv[10*10*k+10*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[8*8*n+8*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_18(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 64; j++){
			 Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_7(fxp input_0[4096], fxp input_1[4096], fxp output[4096]) {
	for (int i = 0; i < 4096; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation16(fxp Input_Activation[4096], fxp Output_Activation[4096]){
	for (int i = 0; i < 4096; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_19(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]){
	loop_for_3_channel_pad_19:
	for (int c = 0; c < 64; c++){
		loop_for_channel_pad_19:
		for (int n = 0; n < 10; n++){
			loop_for_weight_pad_19:
			for (int i = 0; i < 10; i++){
				if (n < 1 || n >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0;
				 else 
					if (i < 1 || i >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0; else output_Pad_Conv[10 * 10 * c + 10 * n + i] = input_Pad_Conv[8 * 8 * c + 8 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_19(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]){
	loop_for_channel2D_19:
	int stride = 1;
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_19:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_19:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_19:
				for (int k = 0; k < 64; k++){
					loop_for_fb_19:
					for (int i = 0; i < 3; i++){
						loop_for_fa_19:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[64*3*3*n+3*3*k+3*i+j])*(Input_Conv[10*10*k+10*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[8*8*n+8*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_19(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 64; j++){
			 Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation17(fxp Input_Activation[4096], fxp Output_Activation[4096]){
	for (int i = 0; i < 4096; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
void Padding_Conv2D_20(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]){
	loop_for_3_channel_pad_20:
	for (int c = 0; c < 64; c++){
		loop_for_channel_pad_20:
		for (int n = 0; n < 10; n++){
			loop_for_weight_pad_20:
			for (int i = 0; i < 10; i++){
				if (n < 1 || n >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0;
				 else 
					if (i < 1 || i >= 9) output_Pad_Conv[10 * 10 * c + 10 * n + i]=0; else output_Pad_Conv[10 * 10 * c + 10 * n + i] = input_Pad_Conv[8 * 8 * c + 8 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_20(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]){
	loop_for_channel2D_20:
	int stride = 1;
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_20:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_20:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_20:
				for (int k = 0; k < 64; k++){
					loop_for_fb_20:
					for (int i = 0; i < 3; i++){
						loop_for_fa_20:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[64*3*3*n+3*3*k+3*i+j])*(Input_Conv[10*10*k+10*(i+x*stride)+j+y*stride]);}
					}
				}
				Output_Conv[8*8*n+8*x+y]=s;
			}
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void BatchNorm2D_20(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 64; j++){
			 Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Add_8(fxp input_0[4096], fxp input_1[4096], fxp output[4096]) {
	for (int i = 0; i < 4096; i++){
		output[i] = input_0[i] + input_1[i];
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 void Activation18(fxp Input_Activation[4096], fxp Output_Activation[4096]){
	for (int i = 0; i < 4096; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
