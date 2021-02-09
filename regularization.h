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

#pragma once

#include <vector>

#define NOMINMAX
#include <windows.h>

#include <omp.h>
#include <string>
#include <cassert>
#include "OptFlowPatchMatch.h"
#include "flowIO.h"

template<typename T>
T bilinear(const T* table, int W, int H, float x, float y, int stride = 1) {
	// mirror / clamp
	/*
	if (x < 0) {
		x = -x;
		x = 0;
	} else if (x > W - 1) {
		x = 2*(W - 1) - x;
		x = W - 1;
	}

	if (y < 0) {
		y = -y;
		y = 0;
	}
	else if (y > H - 1) {
		y = 2*(H - 1) - y;
		y = H - 1;
	}
	*/

	int i = std::max(0, std::min((int)y, H-2));
	float fi = std::min(1.f, std::max(0.f, y-i));
	int j = std::max(0, std::min((int)x, W-2));
	float fj = std::min(1.f, std::max(0.f, x-j));

	return (table[(i*W + j)*stride] * (1.f - fi) + table[((i + 1)*W + j)*stride] * fi)*(1. - fj) + (table[(i*W + j + 1)*stride] * (1.f - fi) + table[((i + 1)*W + j + 1)*stride] * fi)*fj;
}

template<typename T>
static inline T sqr(T x) { return x*x; };

template<typename T>
double get_weight(const T* cur_frame, const T* prev_frame, int W, int H, const float* flow, int pix) {

	T otherVal0 = bilinear(prev_frame+0, W, H, flow[pix*2], flow[pix*2+1], 3);
	T otherVal1 = bilinear(prev_frame+1, W, H, flow[pix*2], flow[pix*2+1], 3);
	T otherVal2 = bilinear(prev_frame+2, W, H, flow[pix*2], flow[pix*2+1], 3);

	const double s = 0.05;
	double w = exp(-(sqr(otherVal0 - cur_frame[pix*3]) + sqr(otherVal1 - cur_frame[pix*3+1]) + sqr(otherVal2 - cur_frame[pix*3+2]))/(2.*s*s));

	if ((flow[pix*2] >= W - 2) || (flow[pix*2] <= 2)
		|| (flow[pix*2+1] >= H - 2) || (flow[pix*2+1] <= 2)) {  // going outside of the image area
		w = 0.0000;

		//int x = pix % W;
		//int y = pix / W;
		//std::cout << x << ":" << y << " ";
		//std::cout << flow[pix * 2] << " " << flow[pix * 2 + 1] << std::endl;
	}
	return w;
}

template<typename T>
void gauss_seidel(T* result_init, const T* processed, const T* diag, const T* rhs, const int W, const int H, const int niter) {

	std::vector<T> tmp(W*H*3);
	T *pxA = result_init, *pxB = &tmp[0];

	for (int iter = 0; iter<niter; iter++) {

#pragma omp parallel for
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				for (int k = 0; k < 3; k++) {
					int p = (i*W+j)*3+k;
					const T up = i>0?pxA[p - 3*W]:0.;
					const T down = i<(H-1)?pxA[p + 3*W]:0.;
					const T left = j>0?pxA[p - 3]:0.;
					const T right = j<(W-1)?pxA[p + 3]:0.;

					int laplace = 4;
					if (i == 0 || i == H - 1) laplace--;
					if (j == 0 || j == W - 1) laplace--;
					pxB[p] =  up + down + left + right + rhs[p];
				}
			}
		}

#pragma omp parallel for
		for (int i = 0; i < W*H; i++) {
			double invdiag = 1./diag[i];
			pxB[i*3] *= invdiag;
			pxB[i*3+1] *= invdiag;
			pxB[i*3+2] *= invdiag;
		}

		std::swap(pxA, pxB);
	}

	if (niter % 2 == 0) {
		memcpy(result_init, &tmp[0], W*H*3*sizeof(result_init[0]));
	}

}

template<typename T>
void multiscale_solver(T* result_init, const T* processed, int W, int H, const T* diag, const T* rhs) {

	int nlevels = 5;

	std::vector<T> downscaled_result(W*H*3);
	memcpy(&downscaled_result[0], result_init, W*H*3*sizeof(T));
	for (int i=nlevels; i>=0; i--) {
		int Wdst = W>>i;
		int Hdst = H>>i;
		cimg_library::CImg<T> res_down(result_init, 3, W, H, 1, false); // shared images pose problem for resize
		cimg_library::CImg<T> processed_down(processed, 3, W, H, 1, false);
		cimg_library::CImg<T> diag_down(diag, 1, W, H, 1, false);
		cimg_library::CImg<T> rhs_down(rhs, 3, W, H, 1, false);

		res_down.resize(3, Wdst, Hdst, 1, 3);  // 3: linear ; 2:moving average ; 5: bicubic		
		processed_down.resize(3, Wdst, Hdst, 1, 3);
		diag_down.resize(1, Wdst, Hdst, 1, 3);
		rhs_down.resize(3, Wdst, Hdst, 1, 3);

		gauss_seidel(res_down.data(), processed_down.data(), diag_down.data(), rhs_down.data(), Wdst, Hdst, 50);

		res_down.resize(3, W, H, 1, 3);
		memcpy(result_init, res_down.data(), W*H*3*sizeof(T));
	}
}

long long milliseconds_now() {
	static LARGE_INTEGER s_frequency;
	static BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
	if (s_use_qpc) {
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		return (1000LL * now.QuadPart) / s_frequency.QuadPart;
	}
	else {
		return GetTickCount();
	}
}

template<typename T>
void solve_frame(const T* prevInput, std::vector<float> flowBwd, const T* curInput, const T* curProcessed, const T* prevSolution, T* curSolution, int W, int H, double lambda_t, bool isFirstFrame) {

	if (isFirstFrame) {
		memcpy(curSolution, curProcessed, W*H*3*sizeof(T));
		return;
	}

	long long of_elapsed = 0;

	if (flowBwd.empty()) {
		long long start = milliseconds_now();
		flowBwd.resize(W * H * 2);
		opt_flow_patchmatch<T>(curInput, prevInput, W, H, &flowBwd[0]);
		long long end = milliseconds_now();
		of_elapsed = end - start;
	}
	assert(flowBwd.size() == W * H * 2);

	/*
	std::vector<float> flowBwdOut(W * H * 2);
	double mag = 0.0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			// should be negated!!
			float u = (float) flowBwd[(size_t) (y * W + x) * 2 + 0]; // -x;
			float v = (float) flowBwd[(size_t) (y * W + x) * 2 + 1]; // -y;
			flowBwdOut[(size_t) (y * W + x) * 2 + 0] = u;
			flowBwdOut[(size_t) (y * W + x) * 2 + 1] = v;
			float m = std::sqrt(u * u + v * v);
			mag += m;
		}
	}
	mag = mag / (W * H);
	//std::cout << "Mean mag: " << mag << std::endl;

	std::string index = std::to_string(i);
	index = std::string(6 - index.size(), '0') + index;
	WriteFlowFile(flowBwdOut, W, H, "test_pwcnetraw/frame_" + index + "_bwd.flo");
	i++;
	*/

	long long start = milliseconds_now();
	
	//build RHS and weights
	std::vector<T> rhs(W*H*3, 0.);
	std::vector<T> diag(W*H, 0.);
#pragma omp parallel for
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			double w = lambda_t * get_weight(curInput, prevInput, W, H, &flowBwd[0], i*W+j);
			int laplace = 4;
			if (i == 0 || i == H - 1) laplace--;
			if (j == 0 || j == W - 1) laplace--;

			int pix = i*W+j;
			for (int k = 0; k < 3; k++) {
				int p = pix*3+k;
				const T upProcessed = i>0?curProcessed[p - 3*W]:0.;
				const T downProcessed = i<(H-1)?curProcessed[p + 3*W]:0.;
				const T leftProcessed = j>0?curProcessed[p - 3]:0.;
				const T rightProcessed = j<(W-1)?curProcessed[p + 3]:0.;
				const T backward_color = bilinear(&prevSolution[k], W, H, flowBwd[pix * 2], flowBwd[pix * 2 + 1], 3);
				rhs[p] = laplace*curProcessed[p] - upProcessed - downProcessed - leftProcessed - rightProcessed  + w * backward_color;
			}
			diag[i*W+j] = laplace + w;
		}
	}

	multiscale_solver(curSolution, curProcessed, W, H, &diag[0], &rhs[0]);

	long long end = milliseconds_now();
	long long optimize_elapsed = end - start;
	std::cout << "Optical flow time: " << of_elapsed << std::endl;
	std::cout << "Stabilization time: " << optimize_elapsed << std::endl;
}

