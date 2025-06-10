#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "common.h"
#include <fcntl.h>
#include <sys/mman.h>
#define t_inf 15000000
int tol[10]={1,2,3,4,5,6,7,8,9,10};
GraphInfo shapes;
using namespace std;
using namespace std::chrono;
using namespace cv;
unsigned int last_reg=198;
unsigned int axi_size = 0x10000;
off_t axi_pbase = 0xa0000000; /* physical base address */
off_t s_axi_base= 0x8f000000;
uint *axi_vptr;
uint *s_axi_ptr;
int fd;
bool ground_flag=true;
bool free_flag=true;
unsigned int diff_tol=1;
unsigned int offset;
unsigned int min_offset;
unsigned int max_offset;
unsigned int width;
unsigned int step;
unsigned int step_glitch;
unsigned int num_glitch;
unsigned int limit;
unsigned int f_prog;

unsigned int g_inference;
unsigned int f_inference;
float iou_thresh=0.25;
/*Files to store results*/
FILE *fpn_raw;

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64,  35, 70, 102, 153, 153, 170, 220, 142, 251,
                    130, 20, 0,  0,   0,   60,  80,  0,   11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

// variables for reading truth mask csv file 

const int MAX_ROWS = 11;
const int MAX_COLS = 524288;
int sample_img;
int data_tmask[MAX_COLS];
float iou[20][2];
int class_count[2][19];

// comparison algorithm for priority_queue
class Compare {
 public:
  bool operator()(const pair<int, Mat>& n1, const pair<int, Mat>& n2) const {
    return n1.first > n2.first;
  }
};

// input video
VideoCapture video;
//const string img_path="input.png";
string img_path;
// flags for each thread
bool is_reading = false;
bool is_running_1 = true;
bool is_running_2 = false;
bool is_displaying = false;

queue<pair<int, Mat>> read_queue;  // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare>
    display_queue;        // display queue
mutex mtx_read_queue;     // mutex of read queue
mutex mtx_display_queue;  // mutex of display queue
int read_index = 0;       // frame index of input video
int display_index = 0;    // frame index to display


void mask_reader(int sample)
{
    ifstream file("mask_512u.csv");
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        //return 1;
    }
    string line;
    int row = 0;
    // Store the CSV data from the CSV file to the 2D array
    while (getline(file, line) && row < MAX_ROWS) {
        stringstream ss(line);
        string cell;
        if(row==sample){
        	int col = 0;
        while (getline(ss, cell, ',') && col < MAX_COLS) {
            data_tmask[col] = stoi(cell);
            //cout << stoi(cell) << " ";
            col++;
            }
        
        }
        
        row++;
    }
    // close the file after read opeartion is complete
    file.close();
}

/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(vart::Runner* runner, bool& is_running) {
  // init out data
  float mean[3] = {103.53, 116.28, 123.675};
  float scale[3] = {0.017429,0.017507,0.01712475};
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
  int batch = inputTensors[0]->get_shape().at(0);
  int8_t* result = new int8_t[shapes.outTensorList[0].size * batch];
  int8_t* result_free = new int8_t[shapes.outTensorList[0].size * batch];
  int8_t* imageInputs = new int8_t[shapes.inTensorList[0].size * batch];
  printf("Output size: %d\n",shapes.outTensorList[0].size);

  /* ----------- Glitch parameters setup ---------------*/
  /* set pointer for axi */
  if ((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) {
    printf("Access memory error");
    //return(0);
  }
  axi_vptr = (uint *)mmap(NULL, axi_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, axi_pbase);
  s_axi_ptr =   (uint *)mmap(NULL, axi_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, s_axi_base);
  // reading true mask 
  mask_reader(sample_img);

  while (is_running) {
    // Get an image from read queue
    int index;
    Mat img;
    img = imread(img_path, IMREAD_COLOR);

    // get in/out tensor
    auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());
    auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);

    // get tensor shape info
    int outHeight = shapes.outTensorList[0].height;
    int outWidth = shapes.outTensorList[0].width;
    int inHeight = shapes.inTensorList[0].height;
    int inWidth = shapes.inTensorList[0].width;

    // image pre-process
    Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
    resize(img, image2, Size(inWidth, inHeight), 0, 0, INTER_LINEAR);

    for (int h = 0; h < inHeight; h++) {
      for (int w = 0; w < inWidth; w++) {
        for (int c = 0; c < 3; c++) {
          imageInputs[h * inWidth * 3 + w * 3 + 2-c] = (int8_t)(
              std::max(std::min((((float)image2.at<Vec3b>(h, w)[c] - mean[c]) * scale[c])*input_scale, 127.0f), -128.0f));
        }
      }
    }

    // tensor buffer prepare
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, inputTensors[0].get()));
    outputs.push_back(
        std::make_unique<CpuFlatTensorBuffer>(result, outputTensors[0].get()));

    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    bool f_csv;
    /*run*/
    ground_flag = true;
    for (unsigned int k = 0; k < limit; k++) {
      /* Setting glitch offset and width*/
      /* integers number are just a reference */
      
      if(k%f_prog==0) printf("Progress: %u %\n",k/f_prog);
      for( unsigned int jj=0;jj<num_glitch;jj++)
      {
          axi_vptr[jj*2]=min_offset + k*step + jj*step_glitch;
          axi_vptr[jj*2+1]=axi_vptr[jj*2]+width;
          //fprintf(fpn_report,"%d,",axi_vptr[jj*2]);
      }
      
      for(int ii=2*num_glitch;ii<last_reg;ii++)
      {
	      axi_vptr[ii]=t_inf;
      }

      int val_compare[10]={0,0,0,0,0,0,0,0,0,0};
      int px_diff[10]={0,0,0,0,0,0,0,0,0,0};
      uint8_t posit_map[512][1024];
      int seg_diff=0;
      memset(class_count, 0, sizeof(class_count));
      for (unsigned int j = 0; j < 2; j++) {
        fprintf(fpn_raw,"%d,",min_offset + k*step);
        if (j % 2 == 0) {
          int status = system("echo 0 > /sys/class/gpio/gpio78/value");
          f_csv = false;
        } else {
          int status = system("echo 1 > /sys/class/gpio/gpio78/value");
          f_csv = true;
          ground_flag = false;
        }
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
      runner->wait(job_id.first, -1);
      //printf("%u-%u\n",axi_vptr[198],axi_vptr[199]);
      if(j==0) g_inference=axi_vptr[last_reg];
      if(j==1) f_inference=axi_vptr[last_reg];
      
      
      for (int row = 0; row < outHeight; row++) {
        for (int col = 0; col < outWidth; col++) {
          int i = row * outWidth * 64 + col * 64;
          if(free_flag){
            result_free[i]=result[i];
            result_free[i+1]=result[i+1];
            result_free[i+2]=result[i+2];
            result_free[i+3]=result[i+3];
            result_free[i+4]=result[i+4];
            result_free[i+5]=result[i+5];
            result_free[i+6]=result[i+6];
            result_free[i+7]=result[i+7];
            result_free[i+8]=result[i+8];
            result_free[i+9]=result[i+9];
            result_free[i+10]=result[i+10];
            result_free[i+11]=result[i+11];
            result_free[i+12]=result[i+12];
            result_free[i+13]=result[i+13];
            result_free[i+14]=result[i+14];
            result_free[i+15]=result[i+15];
            result_free[i+16]=result[i+16];
            result_free[i+17]=result[i+17];
            result_free[i+18]=result[i+18];
            result_free[i+19]=result[i+19];
            result_free[i+20]=result[i+20];
            result_free[i+21]=result[i+21];
            result_free[i+22]=result[i+22];
            result_free[i+23]=result[i+23];
            result_free[i+24]=result[i+24];
            result_free[i+25]=result[i+25];
            result_free[i+26]=result[i+26];
            result_free[i+27]=result[i+27];
            result_free[i+28]=result[i+28];
            result_free[i+29]=result[i+29];
            result_free[i+30]=result[i+30];
            result_free[i+31]=result[i+31];
            result_free[i+32]=result[i+32];
            result_free[i+33]=result[i+33];
            result_free[i+34]=result[i+34];
            result_free[i+35]=result[i+35];
            result_free[i+36]=result[i+36];
            result_free[i+37]=result[i+37];
            result_free[i+38]=result[i+38];
            result_free[i+39]=result[i+39];
            result_free[i+40]=result[i+40];
            result_free[i+41]=result[i+41];
            result_free[i+42]=result[i+42];
            result_free[i+43]=result[i+43];
            result_free[i+44]=result[i+44];
            result_free[i+45]=result[i+45];
            result_free[i+46]=result[i+46];
            result_free[i+47]=result[i+47];
            result_free[i+48]=result[i+48];
            result_free[i+49]=result[i+49];
            result_free[i+50]=result[i+50];
            result_free[i+51]=result[i+51];
            result_free[i+52]=result[i+52];
            result_free[i+53]=result[i+53];
            result_free[i+54]=result[i+54];
            result_free[i+55]=result[i+55];
            result_free[i+56]=result[i+56];
            result_free[i+57]=result[i+57];
            result_free[i+58]=result[i+58];
            result_free[i+59]=result[i+59];
            result_free[i+60]=result[i+60];
            result_free[i+61]=result[i+61];
            result_free[i+62]=result[i+62];
            result_free[i+63]=result[i+63];
          }
      
          if(row==outHeight-1 && col==outWidth-1) {
            fprintf(fpn_raw,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",result[i],result[i+1],result[i+2],result[i+3],result[i+4],result[i+5],result[i+6],result[i+7],result[i+8],result[i+9],result[i+10],result[i+11],result[i+12],result[i+13],result[i+14],result[i+15],result[i+16],result[i+17],result[i+18],result[i+19],result[i+20],result[i+21],result[i+22],result[i+23],result[i+24],result[i+25],result[i+26],result[i+27],result[i+28],result[i+29],result[i+30],result[i+31],result[i+32],result[i+33],result[i+34],result[i+35],result[i+36],result[i+37],result[i+38],result[i+39],result[i+40],result[i+41],result[i+42],result[i+43],result[i+44],result[i+45],result[i+46],result[i+47],result[i+48],result[i+49],result[i+50],result[i+51],result[i+52],result[i+53],result[i+54],result[i+55],result[i+56],result[i+57],result[i+58],result[i+59],result[i+60],result[i+61],result[i+62],result[i+63]);
          }
          else{
            fprintf(fpn_raw,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",result[i],result[i+1],result[i+2],result[i+3],result[i+4],result[i+5],result[i+6],result[i+7],result[i+8],result[i+9],result[i+10],result[i+11],result[i+12],result[i+13],result[i+14],result[i+15],result[i+16],result[i+17],result[i+18],result[i+19],result[i+20],result[i+21],result[i+22],result[i+23],result[i+24],result[i+25],result[i+26],result[i+27],result[i+28],result[i+29],result[i+30],result[i+31],result[i+32],result[i+33],result[i+34],result[i+35],result[i+36],result[i+37],result[i+38],result[i+39],result[i+40],result[i+41],result[i+42],result[i+43],result[i+44],result[i+45],result[i+46],result[i+47],result[i+48],result[i+49],result[i+50],result[i+51],result[i+52],result[i+53],result[i+54],result[i+55],result[i+56],result[i+57],result[i+58],result[i+59],result[i+60],result[i+61],result[i+62],result[i+63]);
          }
        

        }
      }
      free_flag=false;
      

     
 
      }

      printf("Inference cycles fault-free: %d\n",g_inference );
      printf("Inference cycles faulty: %d\n",f_inference );

    }

    inputsPtr.clear();
    outputsPtr.clear();
    inputs.clear();
    outputs.clear();
    is_running = false;
  }
  
  delete imageInputs;
  delete result;
}




/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool& is_reading) {
  while (is_reading) {
    Mat img;
    if (read_queue.size() < 30) {
      if (!video.read(img)) {
        cout << "Finish reading the video." << endl;
        is_reading = false;
        break;
      }
      mtx_read_queue.lock();
      read_queue.push(make_pair(read_index++, img));
      mtx_read_queue.unlock();
    } else {
      usleep(20);
    }
  }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool& is_displaying) {
  while (is_displaying) {
    mtx_display_queue.lock();
    if (display_queue.empty()) {
      if (is_running_1 || is_running_2) {
        mtx_display_queue.unlock();
        usleep(20);
      } else {
        is_displaying = false;
        break;
      }
    } else if (display_index == display_queue.top().first) {
      // Display image
      imshow("Segmentaion @Xilinx DPU", display_queue.top().second);
      display_index++;
      display_queue.pop();
      mtx_display_queue.unlock();
      if (waitKey(1) == 'q') {
        is_reading = false;
        is_running_1 = false;
        is_running_2 = false;
        is_displaying = false;
        break;
      }
    } else {
      mtx_display_queue.unlock();
    }
  }
}

/**
 * @brief Entry for running Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char** argv) {
  // Check args
  if (argc != 10) {
        cout << "Usage: " << argv[0] << " <model_name>" << " <image_path>" << " <min offset>"<<" <max offset>" <<" <max width>" <<" <step>" <<" <sample>" <<" <step glitch>" <<" <num glitches>" << endl;
        return -1;
  }

  string raw_path="/run/media/sda1/seg_records/enet_inner_layer_";
  fpn_raw = fopen((raw_path+argv[7]+"_"+argv[3]+"_"+argv[4]+"_"+argv[5]+"_"+argv[9]+".csv").c_str(),"a"); /*Check this to set file name*/

  min_offset=atoi(argv[3])*100;
  max_offset=atoi(argv[4])*100;
  step=atoi(argv[6]);
  num_glitch=atoi(argv[9]);
  step_glitch=atoi(argv[8]);

  //limit=atoi(argv[8]);
  limit=(max_offset-min_offset)/step;
  f_prog=limit/100;
  width=atoi(argv[5]);
  sample_img=atoi(argv[7]);
  // Initializations
  
  //string file_name = argv[1];
  img_path=argv[2];

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "segmentation should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  // create runner
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  // in/out tensors
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();

  // get in/out tensor shape
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  // Run tasks
  array<thread, 3> threads = {
      thread(Read, ref(is_reading)),
      thread(runSegmentation, runner.get(), ref(is_running_1)),
      thread(Display, ref(is_displaying))};

  for (int i = 0; i < 3; ++i) {
    threads[i].join();
  }

  //video.release();
  printf("Finished \n");
  fclose(fpn_raw);
  return 0;
}

