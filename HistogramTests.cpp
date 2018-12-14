#include "HistogramTests.h"
#include "AccurateTimers.h"
#include "UtilityFunctions.h"
#include "lodepng.h"
#include "CPUHistogram.h"
#include "CUDAHistogram.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <array>
#include <memory>

using namespace std;
using namespace Tests;
using namespace Utils;
using namespace UtilsCUDA;
using namespace Utils::AccurateTimers;
using namespace Utils::CPUParallelism;
using namespace Utils::UnitTests;
using namespace Utils::UtilityFunctions;
namespace 
{
  inline void initializeImg (uint8_t *in, uint32_t w, uint32_t ch, uint8_t val, uint32_t i)
  {
      for (uint32_t j = 0; j < w; j++)
      {
        for (uint32_t k = 0; k < ch; k++) 
        {
          in[i * w * ch + j * ch + k] = val;
        }
      }
    
  }
  inline void setPixel(uint8_t* img, uint32_t x, uint32_t y, uint8_t* color, uint32_t ch, uint32_t w)
  {
    for (uint32_t k = 0; k < ch; k++) 
    {
      img[y * w * ch + x * ch + k] = color[k];
    }
  
  }
  /*
   * Algorithm for line drawing (DDA method) is adapted from URL : https://www.tutorialspoint.com/computer_graphics/pdf/line_generation_algorithm.pdf
   */
  void drawLine(uint8_t* img, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,  uint8_t* color, uint32_t ch, uint32_t w, uint32_t h )
  {

    int dx = abs(x1-x0);
    int dy = abs(y1-y0);
    int err = (dx>dy ? dx : dy); 
    float xs = dx/(float) err;
    float ys = dy/(float) err;

    float x = x0, y = y0;

    for(int i = 0; i <= err; i++)
    {
      if ((uint32_t)x < w && (uint32_t)y < h)
      {
        setPixel(img, (uint32_t)x, (uint32_t)y, color, ch, w);
        x += xs;
        y += ys;
      }
    }
  }
  void plotHistogram(uint32_t* histMap, uint32_t h, uint32_t w, uint32_t numBins, uint32_t ch, string outFilename)
  {
    uint32_t* min_r = std::min_element(histMap, histMap + 256);
    uint32_t* min_g = std::min_element(histMap + 256, histMap + 512);
    uint32_t* min_b = std::min_element(histMap + 512, histMap + 768);
    uint32_t* max_r = std::max_element(histMap, histMap + 256);
    uint32_t* max_g = std::max_element(histMap + 256, histMap + 512);
    uint32_t* max_b = std::max_element(histMap + 512, histMap + 768);
    auto histMapf  = unique_ptr<float[]>(new float[ch * numBins]);
    
    auto histPlotImg = unique_ptr<uint8_t[]>(new uint8_t[w * h * ch ]);
    parallelFor(0, h, [&](size_t i)
    {
      initializeImg (histPlotImg.get(), w, ch, 0, i);
    });

    parallelFor(0, 256, [&](size_t i)
    {
      histMapf[i] = ((histMap[i] - *min_r)/static_cast<float>(*max_r - *min_r)) * (h);
      histMapf[i + 256] = ((histMap[i + 256] - *min_g)/static_cast<float>(*max_g - *min_g)) * (h);
      histMapf[i + 512] = ((histMap[i + 512] - *min_b)/static_cast<float>(*max_b - *min_b)) * (h);
    });

    uint32_t binWidth = std::round(w/numBins);     
    uint8_t color[][3] = { {255,  0,   0}, 
                           {0,  255,   0}, 
                           {0,    0, 255}};
    
    for (uint32_t i = 1; i < numBins; i++) 
    {
      drawLine(histPlotImg.get(), (uint32_t)binWidth * (i-1), h - static_cast<uint32_t> (round(histMapf[i-1])), (uint32_t)binWidth * i,h - static_cast<uint32_t> (round( histMapf[i])), color[0], ch, w, h);
      drawLine(histPlotImg.get(),(uint32_t) binWidth * (i-1), h - static_cast<uint32_t> (round( histMapf[i-1 + 256])),(uint32_t) binWidth * i, h - static_cast<uint32_t> (round(histMapf[i + 256])), color[1], ch, w, h);
      drawLine(histPlotImg.get(), (uint32_t)binWidth * (i-1), h - static_cast<uint32_t> (round(histMapf[i-1 + 512])),(uint32_t) binWidth * i, h - static_cast<uint32_t> (round( histMapf[i + 512])), color[2], ch, w, h);
    } 
 
    uint32_t error = lodepng::encode(outFilename, histPlotImg.get(), w, h, LodePNGColorType::LCT_RGB);
    if (error)
    {
      DebugConsole_consoleOutLine("Lodepng encoder error  ", error, ": ", lodepng_error_text(error));
      EXPECT_TRUE(false);
    }
  }

}

HistogramTest01__SingleParallelExec_Class::HistogramTest01__SingleParallelExec_Class() noexcept
{
  vector<uint8_t> img;
  const string currentPath = StdReadWriteFileFunctions::getCurrentPath();
  const string imgFilename = string(currentPath + "/" + "Assets" + "/" + "alps.png");

  uint32_t imgWidth  = 0;
  uint32_t imgHeight = 0;
  uint32_t error     = lodepng::decode(img, imgWidth, imgHeight, imgFilename, LodePNGColorType::LCT_RGB);
  if (error)
  {
    DebugConsole_consoleOutLine("Lodepng decode error ", error, ": ", lodepng_error_text(error), " for file: ", imgFilename, "\n Aborting the test case execution.");
    EXPECT_TRUE(false);
    return;
  }

  uint32_t ch      = 3;
  uint32_t numBins = 256;

  const CUDADriverInfo gpuInfo;
  const auto histMap = unique_ptr<uint32_t[]>(new uint32_t[ch * numBins]);
  const auto gpuHistMap = unique_ptr<uint32_t[]>(new uint32_t[ch * numBins]);

  //CPU computation of histogram (Includes both Single Core and MultiCore implementation
  CPUHistogram cpuhist(&img[0], imgWidth, imgHeight, ch, numBins, histMap.get());
  EXPECT_TRUE(cpuhist.computeHist());

  //GPU computation of histogram
  //Arguments passed : Input Image Data & its dimension, Output buffer & number of Bins, Reference Data used for verifying. the computed GPU histogram 
  CUDAHistogramCompute gpuhist(gpuInfo, 0, &img[0],imgWidth, imgHeight, ch, numBins, gpuHistMap.get(), histMap.get());
  gpuhist.initializeGPUMemory();
  gpuhist.performGPUComputing();
  gpuhist.retrieveGPUResults();
  EXPECT_TRUE(gpuhist.verifyComputingResults());
  gpuhist.releaseGPUComputingResources();

  //Plotting the output for both GPU and CPU computed histograms - Image Dimension : Width = 800, Height = 400
  string outFilename = string(currentPath + "/" + "Assets" + "/" + "hist_alps_cpu.png");
  plotHistogram(histMap.get(), 400, 800, numBins, ch, outFilename);
  outFilename = string(currentPath + "/" + "Assets" + "/" + "hist_alps_gpu.png");
  plotHistogram(gpuHistMap.get(), 400, 800, numBins, ch, outFilename);
  DebugConsole_consoleOutLine("Summary of Histogram Algorithm Execution");
  DebugConsole_consoleOutLine("Input Image Name :", imgFilename , " Dimension : Height - ", imgHeight, ", Width - ", imgWidth , " Multi channel");
  DebugConsole_consoleOutLine("Single Core CPU Implementation Exec Time :\t", cpuhist.getMeanExecutionTime(0), " msecs");
  DebugConsole_consoleOutLine("MultiCore CPU Implementation Exec Time   :\t", cpuhist.getMeanExecutionTime(1), " msecs");
  DebugConsole_consoleOutLine("Many Core GPU Implementation Exec Time   :\t", gpuhist.getMeanExecutionTime(1), " msecs");
  DebugConsole_consoleOutLine("Histogram Algorithm impl MultiCore CPU Implementation is faster by :\t", cpuhist.getMeanExecutionTime(0)/cpuhist.getMeanExecutionTime(1), "x times in comparison with SingleCore CPU implementation");
  DebugConsole_consoleOutLine("Histogram Algorithm impl Many Core GPU Implementation is faster by :\t", cpuhist.getMeanExecutionTime(1)/gpuhist.getMeanExecutionTime(1), "x times in comparison with MultiCore CPU implementation");
}

TEST(HistogramTest01__SingleParallelExec_Class, ExecSingleManyCore)
{
  HistogramTest01__SingleParallelExec_Class test01;
}
//Entry point for Hist exec.
int main(int argc, char* argv[])
{
#ifdef GPU_FRAMEWORK_DEBUG
  DebugConsole::setUseLogFile(true);
  DebugConsole::setLogFileName("HistUnitTests.log");
#endif // GPU_FRAMEWORK_DEBUG

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
