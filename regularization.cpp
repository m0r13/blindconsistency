// C++ Implementation of: Blind Video Temporal Consistency
// http://liris.cnrs.fr/~nbonneel/consistency/
// if you use this code, please cite:
// 	
// @article{ BTSSPP15,
// author ={ Nicolas Bonneel and James Tompkin and Kalyan Sunkavalli
// and Deqing Sun and Sylvain Paris and Hanspeter Pfister },
// title ={ Blind Video Temporal Consistency },
// journal ={ ACM Transactions on Graphics(Proceedings of SIGGRAPH Asia 2015) },
// volume ={ 34 },
// number ={ 6 },
// year ={ 2015 },
// }

// Copyright (C) 2016  Nicolas Bonneel
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.If not, see <http://www.gnu.org/licenses/>.


// IMPORTANT: for ffmpeg to work correctly, increase the stack size at link time (otherwise, will crash).
// this is a lightweight reimplementation. 
// Only implements the PatchMatch correspondence field ; 
// if you want to use an optical flow, we used : http://people.seas.harvard.edu/~dqsun/publication/2014/ijcv_flow_code.zip  It is in matlab, but can be easily interfaced using system calls.


#include <vector>

#include "regularization.h"
#include "FFGrab.h"
#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>
#include <cassert>

using namespace std;


std::string extract_filename(std::string &path) {
	std::string filename;
	size_t pos = path.find_last_of("\\");
	if (pos != std::string::npos)
		filename.assign(path.begin() + pos + 1, path.end());
	else
		filename = path;
	return filename;
}

std::string extract_fileext(std::string &path) {
	std::string fileext;
	std::string filename = extract_filename(path);
	size_t pos = filename.find_last_of(".");
	if (pos != std::string::npos)
		fileext.assign(filename.begin() + pos + 1, filename.end());
	else
		fileext = "";
	return fileext;
}

int i = 0;

template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
	assert(!(hi < lo));
	return (v < lo) ? lo : (hi < v) ? hi : v;
}

std::vector<float> readFlow(std::string path, int W, int H) {
	std::vector<float> flow;
	ReadFlowFile(flow, path);
	assert(flow.size() == (size_t) W * H * 2);

	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			float& u = flow[(size_t) (y * W + x) * 2 + 0];
			float& v = flow[(size_t) (y * W + x) * 2 + 1];
			//u = clamp(x + u, 0.0f, (float) W - 1);
			//v = clamp(y + v, 0.0f, (float) H - 1);;
			u = x + u;
			v = y + v;
			if (u >= W - 2 || u <= 2 || v >= H - 2 || v <= 2) {
				//u = 512.0;
				//v = 250.0;
				// this is how the patchmatch output looks like
				u = 0.0;
				v = 0.0;
			}
		}
	}
	return flow;
}

int main(int argc, const char* argv[]) {

	
	std::string infile(argv[1]);
	std::string inflow(argv[2]);
	std::string processedfile(argv[3]);
	float lambdaT = atof(argv[4]);
	std::string outfile(argv[5]);
	int nbframes = std::atoi(argv[6]); // max # frames to process

	VideoStreamer<float> *instreamer;
	VideoStreamer<float> *processedstreamer;
	VideoRecorder<float> *outputsRec;

	int W, H;
	if (argc>7) {
		W = atoi(argv[6]);
		H = atoi(argv[7]);
	}

	if (extract_fileext(infile).find("yuv")!=string::npos) {
		instreamer = new VideoStreamerYUV<float>(infile, W, H);
	} else {
		if (extract_fileext(infile).find("png")!=string::npos || extract_fileext(infile).find("bmp")!=string::npos ||
			extract_fileext(infile).find("jpg")!=string::npos || extract_fileext(infile).find("tga")!=string::npos) {
			instreamer = new VideoStreamerImage<float>(infile);
		} else {
			instreamer = new VideoStreamerMPG<float>(infile);
			W = instreamer->W;
			H = instreamer->H;
		}
	}
	if (extract_fileext(processedfile).find("yuv")!=string::npos) {
		processedstreamer = new VideoStreamerYUV<float>(processedfile, W, H);
	} else {
		if (extract_fileext(processedfile).find("png")!=string::npos || extract_fileext(processedfile).find("bmp")!=string::npos ||
			extract_fileext(processedfile).find("jpg")!=string::npos || extract_fileext(processedfile).find("tga")!=string::npos) {
			processedstreamer = new VideoStreamerImage<float>(processedfile);
		} else {
			processedstreamer = new VideoStreamerMPG<float>(processedfile);
		}
	}


	
	nbframes = std::min(nbframes, std::min(instreamer->nbframes, processedstreamer->nbframes));

	if (extract_fileext(outfile).find("yuv")!=string::npos) {
		outputsRec = new VideoRecorderYUV<float>(outfile.c_str(), W, H);
	} else {
		if (extract_fileext(outfile).find("png")!=string::npos || extract_fileext(outfile).find("bmp")!=string::npos ||
			extract_fileext(outfile).find("jpg")!=string::npos || extract_fileext(outfile).find("tga")!=string::npos) {
			outputsRec = new VideoRecorderImage<float>(outfile.c_str(), W, H);
		} else {
			outputsRec = new VideoRecorderMPG<float>(outfile.c_str(), W, H);
		}
	}


	std::vector<float> prevInput(W*H*3);
	std::vector<float> curInput(W*H*3);

	std::vector<float> curProcessed(W*H*3);

	std::vector<float> prevSolution(W*H*3);
	std::vector<float> curSolution(W*H*3);

	//std::cout << "Number frames" << nbframes << std::endl;
	//return 0;

	auto flowPath = [inflow](int i) {
		std::string flowIndex = std::to_string(i);
		// zero padding to 6 digits
		flowIndex = std::string(6 - flowIndex.size(), '0') + flowIndex;
		std::string path = inflow + "/frame_" + flowIndex + "_bwd.flo";
		return path;
	};

	for (int i=0; i<nbframes; i++) {

		std::cout<<"processing frame "<<i<<" over "<<nbframes<<std::endl;
		if (!instreamer->get_next_frame(&curInput[0])) break;
		if (!processedstreamer->get_next_frame(&curProcessed[0])) break;

		std::vector<float> flowBwd;
		if (inflow != "-") {
			flowBwd = readFlow(flowPath(i), W, H);
			assert(flowBwd.size() == (size_t) W * H * 2);
		}

		curSolution = curProcessed;
		solve_frame<float>(&prevInput[0], flowBwd, &curInput[0], &curProcessed[0], &prevSolution[0], &curSolution[0], W, H, lambdaT, i==0);		

		prevInput = curInput;
		prevSolution = curSolution;		

		outputsRec->addFrame(&curSolution[0]);

	}
	
	outputsRec->finalize_video();


	return 0;
}


