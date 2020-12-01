#include <kernel_common.hpp>
#include <algorithm>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_samplesort(
          hb_tensor_t* sorted_,
          hb_tensor_t* inp,
          hb_tensor_t* sample_keys,
          hb_tensor_t* sorted_keys,
          hb_tensor_t* splitters,
          hb_tensor_t* buck_sizes,
          int32_t* nproc,
          int32_t* sr
          ) {

    auto hb_res = HBTensor<float>(sorted_);
    auto hb_inp = HBTensor<float>(inp);
    auto hb_sample_keys = HBTensor<float>(sample_keys);
    auto hb_sorted_keys = HBTensor<float>(sorted_keys);
    auto hb_splitters = HBTensor<float>(splitters);
    auto hb_buck_sizes = HBTensor<float>(buck_sizes);
    int32_t hb_nproc = *nproc;
    int32_t hb_sr = *sr; //sampling rate

    if(__bsg_id == 0){
      hb_assert_msg(hb_nproc <= (bsg_tiles_X * bsg_tiles_Y),
                  "desired cores are not present on the hardware");
      hb_assert_msg((hb_nproc*hb_sr) <= hb_inp.numel(),
                  "total sampling elements should be less than total list size");
    }
    
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    int n_sampler = hb_nproc-1;
    int list_size = hb_inp.numel();
    int seed, index, offset, local_chunk_size, local_sample_size;

    int total_sample_size = hb_sr*n_sampler;

    bsg_attr_remote float* res_ptr = (bsg_attr_remote float*)hb_res.data_ptr();
    bsg_attr_remote float* inp_ptr = (bsg_attr_remote float*)hb_inp.data_ptr();
    
    //use one processor/tile
    if (hb_nproc==1) {
      if (__bsg_id==0){
        memcpy(res_ptr,inp_ptr,list_size*sizeof(float));
        std::sort(res_ptr,res_ptr+list_size,std::less<float>());
      }
    }
    else{
      srand(__bsg_id+1);
      //generate sample keys
      local_chunk_size = list_size/n_sampler;
      offset = __bsg_id * hb_sr;
      if(__bsg_id<n_sampler){
        for(int i=offset; i< (offset+hb_sr); i++){
          seed = (__bsg_id * local_chunk_size) + (rand() % local_chunk_size);
          hb_sample_keys(i) = hb_inp(seed);
        }
      }
      g_barrier.sync();

      //sort sample keys
      if(__bsg_id<n_sampler){
        for(int i=offset; i< (offset+hb_sr); i++){
          float mykey = hb_sample_keys(i);
          int myindex=0;
          for (int j = 0; j < total_sample_size; j++) {
            if (hb_sample_keys(j) < mykey) {
              myindex++;
            } else if (hb_sample_keys(j) == mykey && j < i) {
              myindex++;
            } else {
            }
          }
          hb_sorted_keys(myindex) = mykey;
        }
      }
      g_barrier.sync();

      //get (nproc-1) splitters from the keys
      if(__bsg_id<n_sampler){
        hb_splitters(__bsg_id) = hb_sorted_keys(offset+(hb_sr/2));
      }
      g_barrier.sync();

      //find bucket size
      int buck_size=0;
      int c1,c2;
      if(__bsg_id<hb_nproc){
        for (int j = 0; j < list_size; j++) {
          c1=c2=1;
          if(__bsg_id>0 && hb_inp(j)<hb_splitters(__bsg_id-1)) c1=0;
          if(__bsg_id<(hb_nproc-1) && hb_inp(j)>=hb_splitters(__bsg_id)) c2=0;
          if (c1 && c2){
            buck_size++;
          }
        }
        hb_buck_sizes(__bsg_id)=buck_size;
        printf("Bucket size %d, id %d\n",buck_size,__bsg_id);
      }
      g_barrier.sync();

      //find offsets
      int my_offset=0;
      if(__bsg_id<hb_nproc){
        for(int j=0; j<__bsg_id; j++) {
          my_offset+=hb_buck_sizes(j);
        }
      }

      bsg_attr_remote float* bucket = res_ptr + my_offset;
      int iter=0;
      //sort bucket
      if(__bsg_id<hb_nproc){
        for (int j = 0; j < list_size; j++) {
          c1=c2=1;
          if(__bsg_id>0 && hb_inp(j)<hb_splitters(__bsg_id-1)) c1=0;
          if(__bsg_id<(hb_nproc-1) && hb_inp(j)>=hb_splitters(__bsg_id)) c2=0;
          if (c1 && c2){
            bucket[iter] = hb_inp(j);
            iter++;
          }
        }
        //sort bucket
        std::sort(bucket,bucket+buck_size,std::less<float>());
      }
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_samplesort, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int32_t*, int32_t*)

}
