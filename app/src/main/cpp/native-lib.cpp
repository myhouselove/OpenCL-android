#include <jni.h>
#include <stdlib.h>
#include <string>
#include <opencl.h>

void printDeviceWorkInfo(cl_device_id device)
{
    cl_uint nMaxComputeUnits = 0;
    cl_uint nMaxWorkItemDims = 0;
    cl_uint i = 0;
    size_t* nMaxWorkItemSizes = NULL;
    size_t nMaxWorkGroupSize = 0;
    size_t size = 0 ;
    cl_int err ;
    err = clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&nMaxComputeUnits,&size);
    if(err==CL_SUCCESS){
        printf("nMaxComputeUnits=%d\n",nMaxComputeUnits);
    }

    err = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(cl_uint),&nMaxWorkItemDims,&size);
    if(err==CL_SUCCESS){
        printf("nMaxWorkItemDims=%d\n",nMaxWorkItemDims);
        nMaxWorkItemSizes = (size_t*)malloc(sizeof(size_t)*nMaxWorkItemDims);
        err = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(size_t)*nMaxWorkItemDims,nMaxWorkItemSizes,&size);
        if(err==CL_SUCCESS){
            for(i=0;i<nMaxWorkItemDims;i++){
                printf("nMaxWorkItemSizes[%d]=%d\n",i,nMaxWorkItemSizes);
            }
        }
        free(nMaxWorkItemSizes);
    }

    err = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&nMaxWorkGroupSize,&size);
    if(err==CL_SUCCESS){
        printf("nMaxWorkGroupSize=%d\n",nMaxWorkGroupSize);
    }
}

const char* program_src = ""
        "__kernel void vector_add_gpu (__global const float* src_a,\n"
        "   __global const float* src_b,\n"
        "   __global float* res,\n"
        "   const int num)\n"
        "{\n"
        "   int idx = get_global_id(0);\n"
        "   if(idx<num){"
        "       res=src_a+src_b;\n"
        "   }\n"
        "}\n"
;

static const cl_int vect_len = 10000000;

static float* vect_a = NULL ;
static float* vect_b = NULL ;
static float* vect_c = NULL ;

void initVects()
{
    cl_int i;
    vect_a = (float*)malloc(sizeof(float)*vect_len);
    vect_b = (float*)malloc(sizeof(float)*vect_len);
    vect_c = (float*)malloc(sizeof(float)*vect_len);
    for(i=0;i<vect_len;i++){
        *vect_a=(float)rand()/RAND_MAX;
        *vect_b=(float)rand()/RAND_MAX;
        *vect_c=0.0f;
    }
}

void printVects()
{
    cl_int i;
    if(vect_a && vect_b && vect_c){
        printf("######################\n");
        for(i=0;i<4;i++){
            printf("%08d : %f,%f,%f\n",i,vect_a,vect_b,vect_c);
        }
        printf("    ...    \n");
        for(i=vect_len-4;i<vect_len;i++){
            printf("%08d : %f,%f,%f\n",i,vect_a,vect_b,vect_c);
        }
        printf("######################\n");
    }
}

void releaseVects()
{
    if(vect_a){
        free(vect_a);
        vect_a=NULL;
    }
    if(vect_b){
        free(vect_b);
        vect_b=NULL;
    }
    if(vect_c){
        free(vect_c);
        vect_c=NULL;
    }
}

size_t shrRoundUp(size_t f , size_t s)
{
    return (s+f-1)/f*f;
}



void test()
{
    cl_int error = 0 ;
    cl_platform_id platform;
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_mem inbuf_a ;
    cl_mem inbuf_b ;
    cl_mem outbuf_r ;
    const cl_int size = vect_len;
    cl_int i ;
    const size_t mem_size = sizeof(float)*size;
    size_t program_len = strlen(program_src);
    char* build_log;
    size_t log_size;
    size_t local_ws;
    size_t global_ws;
    cl_kernel vector_add_kernel;

    error = clGetPlatformIDs(1,&platform,NULL);
    if(error != CL_SUCCESS){
        printf("get platform id fail !\n");
        exit(1);
    }

    error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,NULL);
    if(error != CL_SUCCESS){
        printf("get gpu device fail !\n");
        exit(1);
    }

    printDeviceWorkInfo(device);

    cl_context_properties properties[]={
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platform,
            0
    };

    // 这里要配置properties
    context = clCreateContext(properties,1,&device,NULL,NULL,&error);
    if(error != CL_SUCCESS){
        printf("create context fail !\n");
        exit(1);
    }

    queue = clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE,&error);
    if(error != CL_SUCCESS){
        printf("create command queue fail !\n");
        exit(1);
    }

    initVects();
    printVects();

    inbuf_a = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,mem_size,vect_a,&error);
    if(error!=CL_SUCCESS){
        printf("create buffer inbuf_a fail !\n");
        exit(1);
    }
    inbuf_b = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,mem_size,vect_b,&error);
    if(error!=CL_SUCCESS){
        printf("create buffer inbuf_b fail !\n");
        exit(1);
    }
    outbuf_r = clCreateBuffer(context,CL_MEM_WRITE_ONLY,mem_size,NULL,&error);
    if(error!=CL_SUCCESS){
        printf("create buffer outbuf_r fail !\n");
        exit(1);
    }

    cl_program program = clCreateProgramWithSource(context,1,&program_src,&program_len,&error);
    if(error!=CL_SUCCESS){
        printf("create program fail !\n");
        exit(1);
    }
    error = clBuildProgram(program,1,&device,NULL,NULL,NULL);
    if(error!=CL_SUCCESS){
        printf("build program fail !\n");
        clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,1024,build_log,&log_size);
        printf("build_log : %s\n",build_log);
        exit(1);
    }

    vector_add_kernel = clCreateKernel(program,"vector_add_gpu",&error);
    if(error!=CL_SUCCESS){
        printf("create kernel fail !\n");
        exit(1);
    }

    error = clSetKernelArg(vector_add_kernel,0,sizeof(cl_mem),&inbuf_a);
    error |= clSetKernelArg(vector_add_kernel,1,sizeof(cl_mem),&inbuf_b);
    error |= clSetKernelArg(vector_add_kernel,2,sizeof(cl_mem),&outbuf_r);
    error |= clSetKernelArg(vector_add_kernel,3,sizeof(cl_int),&size);
    if(error!=CL_SUCCESS){
        printf("set kernel arg fail !\n");
        exit(1);
    }

    local_ws = 256;//我们使用一维的clEnqueueNDRangeKernel，这里local_ws选择nMaxWorkItemSizes=256
    global_ws = shrRoundUp(local_ws,size); //这里是线程总数，应该是local_ws的倍数。
    printf("local_ws=%d,global_ws=%d\n",local_ws,global_ws);

    error = clEnqueueNDRangeKernel(queue,vector_add_kernel,1,NULL,&global_ws,&local_ws,0,NULL,NULL);
    if(error!=CL_SUCCESS){
        printf("enqueue kernel fail !\n");
        exit(1);
    }

    clEnqueueReadBuffer(queue,outbuf_r,CL_TRUE,0,mem_size,vect_c,0,NULL,NULL);
    printVects();

    clReleaseKernel(vector_add_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseMemObject(inbuf_a);
    clReleaseMemObject(inbuf_b);
    clReleaseMemObject(outbuf_r);
    releaseVects();
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_wangmingyong_opencl_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    test();

    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
