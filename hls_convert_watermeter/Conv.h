#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Padding_Conv2D_0(fxp input_Pad_Conv[1024], fxp output_Pad_Conv[1156]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Conv2D_0(fxp Input_Conv[1156],fxp Output_Conv[16384], fxp kernel[144]);
void BatchNorm2D_0(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation0(fxp Input_Activation[16384], fxp Output_Activation[16384]);
void Padding_Conv2D_1(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]);
void Conv2D_1(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]);
void BatchNorm2D_1(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation1(fxp Input_Activation[16384], fxp Output_Activation[16384]);
void Padding_Conv2D_2(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]);
void Conv2D_2(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]);
void BatchNorm2D_2(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]);
void Add_0(fxp input_0[16384], fxp input_1[16384], fxp output[16384]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation2(fxp Input_Activation[16384], fxp Output_Activation[16384]);
void Padding_Conv2D_3(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]);
void Conv2D_3(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]);
void BatchNorm2D_3(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation3(fxp Input_Activation[16384], fxp Output_Activation[16384]);
void Padding_Conv2D_4(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]);
void Conv2D_4(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]);
void BatchNorm2D_4(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]);
void Add_1(fxp input_0[16384], fxp input_1[16384], fxp output[16384]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation4(fxp Input_Activation[16384], fxp Output_Activation[16384]);
void Padding_Conv2D_5(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]);
void Conv2D_5(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]);
void BatchNorm2D_5(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation5(fxp Input_Activation[16384], fxp Output_Activation[16384]);
void Padding_Conv2D_6(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[18496]);
void Conv2D_6(fxp Input_Conv[18496],fxp Output_Conv[16384], fxp kernel[2304]);
void BatchNorm2D_6(fxp Input_BatchNorm[16384], fxp Output_BatchNorm[16384], fxp gamma[16], fxp beta[16], fxp MovMean[16], fxp MovVar[16]);
void Add_2(fxp input_0[16384], fxp input_1[16384], fxp output[16384]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation6(fxp Input_Activation[16384], fxp Output_Activation[16384]);
void Padding_Conv2D_7(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[17424]);
void Conv2D_7(fxp Input_Conv[17424],fxp Output_Conv[8192], fxp kernel[4608]);
void BatchNorm2D_7(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation7(fxp Input_Activation[8192], fxp Output_Activation[8192]);
void Padding_Conv2D_8(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]);
void Conv2D_8(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]);
void Padding_Conv2D_9(fxp input_Pad_Conv[16384], fxp output_Pad_Conv[15376]);
void Conv2D_9(fxp Input_Conv[15376],fxp Output_Conv[8192], fxp kernel[512]);
void BatchNorm2D_8(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
void BatchNorm2D_9(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
void Add_3(fxp input_0[8192], fxp input_1[8192], fxp output[8192]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation8(fxp Input_Activation[8192], fxp Output_Activation[8192]);
void Padding_Conv2D_10(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]);
void Conv2D_10(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]);
void BatchNorm2D_10(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation9(fxp Input_Activation[8192], fxp Output_Activation[8192]);
void Padding_Conv2D_11(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]);
void Conv2D_11(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]);
void BatchNorm2D_11(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
void Add_4(fxp input_0[8192], fxp input_1[8192], fxp output[8192]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation10(fxp Input_Activation[8192], fxp Output_Activation[8192]);
void Padding_Conv2D_12(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]);
void Conv2D_12(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]);
void BatchNorm2D_12(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation11(fxp Input_Activation[8192], fxp Output_Activation[8192]);
void Padding_Conv2D_13(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[10368]);
void Conv2D_13(fxp Input_Conv[10368],fxp Output_Conv[8192], fxp kernel[9216]);
void BatchNorm2D_13(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
void Add_5(fxp input_0[8192], fxp input_1[8192], fxp output[8192]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation12(fxp Input_Activation[8192], fxp Output_Activation[8192]);
void Padding_Conv2D_14(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[9248]);
void Conv2D_14(fxp Input_Conv[9248],fxp Output_Conv[4096], fxp kernel[18432]);
void BatchNorm2D_14(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation13(fxp Input_Activation[4096], fxp Output_Activation[4096]);
void Padding_Conv2D_15(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]);
void Conv2D_15(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]);
void Padding_Conv2D_16(fxp input_Pad_Conv[8192], fxp output_Pad_Conv[7200]);
void Conv2D_16(fxp Input_Conv[7200],fxp Output_Conv[4096], fxp kernel[2048]);
void BatchNorm2D_15(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
void BatchNorm2D_16(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
void Add_6(fxp input_0[4096], fxp input_1[4096], fxp output[4096]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation14(fxp Input_Activation[4096], fxp Output_Activation[4096]);
void Padding_Conv2D_17(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]);
void Conv2D_17(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]);
void BatchNorm2D_17(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation15(fxp Input_Activation[4096], fxp Output_Activation[4096]);
void Padding_Conv2D_18(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]);
void Conv2D_18(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]);
void BatchNorm2D_18(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
void Add_7(fxp input_0[4096], fxp input_1[4096], fxp output[4096]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation16(fxp Input_Activation[4096], fxp Output_Activation[4096]);
void Padding_Conv2D_19(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]);
void Conv2D_19(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]);
void BatchNorm2D_19(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation17(fxp Input_Activation[4096], fxp Output_Activation[4096]);
void Padding_Conv2D_20(fxp input_Pad_Conv[4096], fxp output_Pad_Conv[6400]);
void Conv2D_20(fxp Input_Conv[6400],fxp Output_Conv[4096], fxp kernel[36864]);
void BatchNorm2D_20(fxp Input_BatchNorm[4096], fxp Output_BatchNorm[4096], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
void Add_8(fxp input_0[4096], fxp input_1[4096], fxp output[4096]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Activation18(fxp Input_Activation[4096], fxp Output_Activation[4096]);
