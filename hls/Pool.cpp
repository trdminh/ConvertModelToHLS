void GlobalAveragePool2D_0(fxp input_GlobalAveragePool2D[4096],fxp output_GlobalAveragePool2D[64]){
	int hs = 0;
	for (int i = 0; i < 64; i++){
		fxp avg = 0;
		for (int j = 0; j < 8; j++){
			for (int k = 0; k < 8; k++){
				avg += input_GlobalAveragePool2D[8 * 8 * i + 8 * j + k];
			}
		}
		output_GlobalAveragePool2D[hs] = avg / (8 * 8) ;
		hs++;
	}
}
