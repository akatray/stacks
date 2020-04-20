// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include "./../Layer.hpp"

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Neural Networks Experiment.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Convolutional layer 2d. Filters each channel separately.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		uMAX WIDTH_IN,
		uMAX HEIGHT_IN,
		uMAX DEPTH_IN,
		uMAX RADIUS = 1,
		FnTrans FN_TRANS = FnTrans::RELU,
		sx::FnInitWeights FN_INIT_W = sx::FnInitWeights::NRM_RELU,
		FnOptim FN_OPTIM = FnOptim::ADAM,
		FnErr FN_ERR = FnErr::MSE
	>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class Conv2Parallel :
		public Layer<T>,
		LDOutputs<T, WIDTH_IN*HEIGHT_IN*DEPTH_IN, WIDTH_IN*HEIGHT_IN*DEPTH_IN, true>,
		LDWeights<T, FN_OPTIM, uMAX(((RADIUS*2)+1)*((RADIUS*2)+1))*DEPTH_IN, WIDTH_IN*HEIGHT_IN*DEPTH_IN, WIDTH_IN*HEIGHT_IN*DEPTH_IN, FN_INIT_W>,
		LDBiases<T, WIDTH_IN*HEIGHT_IN*DEPTH_IN, 0, 0, needBufM<T,FN_OPTIM>(), needBufV<T,FN_OPTIM>()>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto SZ_KER_EDGE = uMAX(RADIUS * 2 + 1);
		constexpr static auto SZ_KER = SZ_KER_EDGE * SZ_KER_EDGE;
		constexpr static auto SZ_KER_RDX_MIN = -iMAX(RADIUS);
		constexpr static auto SZ_KER_RDX_MAX = iMAX(RADIUS + 1);

		constexpr static auto LINE_BEG = RADIUS;
		constexpr static auto LINE_END = WIDTH_IN - RADIUS;
		constexpr static auto LINE_LEN = WIDTH_IN - (RADIUS * 2);

		constexpr static auto SZ_BUF_W = SZ_KER * DEPTH_IN;
		constexpr static auto SZ_BUF_B = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_IN = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_OUT = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Generated functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_MC_LAYER_TRIVIAL(Conv2Parallel, SZ_OUT, this->OutTrans, this->Gradient)

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute. Optimized for sequential access.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_EXE final
		{
			memZero(SZ_OUT, this->OutRaw);

			for(auto d = uMAX(0); d < DEPTH_IN; ++d)
			{ 
				auto LnKernel = this->Weights + math::index_c(0, d, SZ_KER);
				for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y)
				{
					auto LnOutRw = this->OutRaw + math::index_c(RADIUS, y, d, WIDTH_IN, HEIGHT_IN);
					for(auto kr = uMAX(0); kr < SZ_KER_EDGE; ++kr)
					{
						auto LnIn = this->Input + math::index_c(0, y - RADIUS + kr, d, WIDTH_IN, HEIGHT_IN);
						for(auto x = uMAX(0); x < LINE_LEN; ++x)
						{
							for(auto w = uMAX(0); w < SZ_KER_EDGE; ++w) LnOutRw[x] += LnIn[x+w] * LnKernel[math::index_c(w, kr, SZ_KER_EDGE)];
						}
					}

					auto LnBias = this->Biases + math::index_c(RADIUS, y, d, WIDTH_IN, HEIGHT_IN);
					auto LnOutTr = this->OutTrans + math::index_c(RADIUS, y, d, WIDTH_IN, HEIGHT_IN);
					for(auto x = uMAX(0); x < LINE_LEN; ++x)
					{
						LnOutRw[x] += LnBias[x];
						LnOutTr[x] = transfer<T,FN_TRANS>(LnOutRw[x]);
					}
				}
			}

			SX_MC_LAYER_NEXT_EXE;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate. Optimized for memory order.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_FIT final
		{
			memZero(SZ_IN, this->Gradient);

			if(this->Front)
			{
				// For each kernel.
				auto PtrFrontGradient = this->Front->gradient();
				auto PtrTrDerSrc = this->OutRaw;
				if constexpr(!needRaw<T,FN_TRANS>()) PtrTrDerSrc = this->OutTrans;
				for(auto d = uMAX(0); d < DEPTH_IN; ++d)
				{ 
					// For each channel.
					const auto OffKernel = math::index_c(0, d, SZ_KER);
					auto LnKernel = this->Weights + OffKernel;
					auto LnKernelDlt = this->WeightsDlt + OffKernel;
					for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y)
					{
						// Calculate derivatives of horizontal line.
						const auto OffOut = math::index_c(0, y, d, WIDTH_IN, HEIGHT_IN);
						auto LnOutUn = PtrTrDerSrc + OffOut;
						T LnDerTrans[WIDTH_IN];
						memCopy(WIDTH_IN, LnDerTrans, PtrFrontGradient + OffOut);
						for(auto x = 0; x < WIDTH_IN; ++x) LnDerTrans[x] *= transferDer<T,FN_TRANS>(LnOutUn[x]);
						
						// Update bias deltas of horizontal line.
						auto LnBiasDlt = this->BiasesDlt + OffOut;
						for(auto x = 0; x < WIDTH_IN; ++x) LnBiasDlt[x] += LnDerTrans[x];

						// For each kernel row of horizontal line.
						for(auto kr = uMAX(0); kr < SZ_KER_EDGE; ++kr)
						{
							// For each pixel in horizontal line.
							const auto OffIn = math::index_c(0, y - RADIUS + kr, d, WIDTH_IN, HEIGHT_IN);
							auto LnIn = this->Input + OffIn;
							auto LnGrad = this->Gradient + OffIn;
							for(auto x = 0; x < LINE_LEN; ++x)
							{
								// For each weight in kernel row.
								const auto IdxOut = x+RADIUS;
								for(auto w = uMAX(0); w < SZ_KER_EDGE; ++w)
								{
									const auto IdxKer = math::index_c(w, kr, SZ_KER_EDGE);
									const auto IdxIn = x+w;
									LnKernelDlt[IdxKer] += LnIn[IdxIn] * LnDerTrans[IdxOut];
									LnGrad[IdxIn] += LnKernel[IdxKer] * LnDerTrans[IdxOut];
								}
							}
						}
					}
				}
			}

			else throw fx::Error("sx"s, "Conv2Parallel"s, "fit"s, 0, "Can't be last layer!"s);

			SX_MC_LAYER_NEXT_FIT;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get output error in respect to argument.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplErr.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset delta parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplReset.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply optimizations and update parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplApply.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store parameters to stream.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplStore.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load parameters from stream.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplLoad.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Multi threading utility.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplExchange.hpp"
	};
}
