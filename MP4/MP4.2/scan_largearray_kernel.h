
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
// MP4.2 - You can use any other block size you wish.
#define BLOCK_SIZE 256

void preallocBlockSums(int num_elements);
void prescanArray(float *outArray, float *inArray, int numElements);

#endif // _PRESCAN_CU_
