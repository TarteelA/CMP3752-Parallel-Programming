//Histogram Implementation
kernel void hist(global const uchar* a, global int* b) {
	int ID = get_global_id(0);
	atomic_inc(&b[a[ID]]);
}

//Histogram Cumulative Implementation
kernel void hist_cum(global const int* a, global int* b) {
	int ID = get_global_id(0);
	int n = get_global_size(0);
	for (int i = ID + 1; i < n; i++)
		atomic_add(&b[i], a[ID]);
}

//LUT
kernel void LUT(global const int* a, global int* b) {
	int ID = get_global_id(0);
	b[ID] = a[ID] * (double)255 / a[255];
}

//OpenCL Simple Kernel copies pixels from a to b
kernel void ReProject(global const uchar* a, global const int* LUT, global uchar* b) {
	int ID = get_global_id(0);
	b[ID] = LUT[a[ID]];
}