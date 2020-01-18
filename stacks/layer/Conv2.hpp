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
	// Convolutional layer 2d.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		uMAX WIDTH_IN,
		uMAX HEIGHT_IN,
		uMAX DEPTH_IN,
		uMAX KERNELS = 1,
		uMAX RADIUS = 1,
		FnTrans FN_TRANS = FnTrans::PRELU,
		FnOptim FN_OPTIM = FnOptim::MOMENTUM,
		FnErr FN_ERR = FnErr::MSE
	>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class Conv2 :
		public Layer<T>,
		LDOutputs<T, WIDTH_IN * HEIGHT_IN * KERNELS, WIDTH_IN * HEIGHT_IN * DEPTH_IN, true>,
		LDWeights<T, uMAX(((RADIUS*2)+1)*((RADIUS*2)+1)) * KERNELS * DEPTH_IN, 0, 0, needBufM<T,FN_OPTIM>(), needBufV<T,FN_OPTIM>()>,
		LDBiases<T, WIDTH_IN * HEIGHT_IN * KERNELS, 0, 0, needBufM<T,FN_OPTIM>(), needBufV<T,FN_OPTIM>()>
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
		constexpr static auto LINE_LEN = WIDTH_IN - RADIUS * 2;

		constexpr static auto SZ_BUF_W = SZ_KER * KERNELS * DEPTH_IN;
		constexpr static auto SZ_BUF_B = WIDTH_IN * HEIGHT_IN * KERNELS;
		constexpr static auto SZ_IN = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_OUT = WIDTH_IN * HEIGHT_IN * KERNELS;
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Generated functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_MC_LAYER_TRIVIAL(Conv2, SZ_OUT, this->OutTrans, this->Gradient)

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute layer.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_EXE final
		{
			memZero(SZ_OUT, this->OutRaw);

			for(auto k = uMAX(0); k < KERNELS; ++k)
			{ 
				auto LnWeights = this->Weights + math::index_c(0, k, SZ_KER);
				for(auto d = uMAX(0); d < DEPTH_IN; ++d) { for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y)
				{
					auto LnOut = this->OutRaw + math::index_c(RADIUS, y, k, WIDTH_IN, HEIGHT_IN);
					auto LnInRow = this->Input + math::index_c(0, y - RADIUS, d, WIDTH_IN, HEIGHT_IN);
					for(auto kr = uMAX(0); kr < SZ_KER_EDGE; ++kr)
					{
						for(auto x = LINE_BEG; x < LINE_END; ++x)
						{
							for(auto w = uMAX(0); w < SZ_KER_EDGE; ++w) *(LnOut) += *(LnInRow+w) * *(LnWeights+w);
							LnInRow++;
							LnOut++;
						}

						LnInRow += WIDTH_IN - LINE_LEN;
						LnWeights += SZ_KER_EDGE;
						LnOut -= LINE_LEN;
					}

					LnWeights -= SZ_KER;
				}}
			}

			for(auto o = uMAX(0); o < SZ_OUT; ++o) this->OutTrans[o] = transfer<T,FN_TRANS>(this->OutRaw[o] + this->Biases[o]);

			SX_MC_LAYER_NEXT_EXE;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_FIT final
		{
			memZero(SZ_IN, this->Gradient);

			for(auto k = uMAX(0); k < KERNELS; ++k)
			{ 
				auto LnWeights = this->Weights + math::index_c(0, k, SZ_KER);
				auto LnWeightsDlt = this->WeightsDlt + math::index_c(0, k, SZ_KER);
				for(auto d = uMAX(0); d < DEPTH_IN; ++d) { for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y)
				{
					const auto o = math::index_c(RADIUS, y, k, WIDTH_IN, HEIGHT_IN);
					auto LnOut = this->OutRaw + math::index_c(RADIUS, y, k, WIDTH_IN, HEIGHT_IN);
					auto LnInRow = this->Input + math::index_c(0, y - RADIUS, d, WIDTH_IN, HEIGHT_IN);
					auto LnGradRow = this->Gradient + math::index_c(0, y - RADIUS, d, WIDTH_IN, HEIGHT_IN);
					T DerCache[WIDTH_IN];

					if(this->Front)
					{
						memCopy(LINE_LEN, DerCache + LINE_BEG, this->Front->gradient() + o);
					}

					else
					{
						auto LnTarget = _Target + o;
						auto LnOutTrans = this->OutTrans + o;
						for(auto x = LINE_BEG; x < LINE_END; ++x)
						{
							DerCache[x] = errorDer<T,FN_ERR>(*LnTarget, *LnOutTrans);
							LnTarget++;
							LnOutTrans++;
						}
					}

					auto LnOutTrans = this->OutTrans + o;
					auto LnOutRaw = this->OutRaw + o;
					for(auto x = LINE_BEG; x < LINE_END; ++x)
					{
						if constexpr(needRaw<T,FN_TRANS>())
						{
							DerCache[x] *= transferDer<T,FN_TRANS>(*LnOutTrans, *LnOutRaw);
							LnOutTrans++;
							LnOutRaw++;
						}

						else
						{
							DerCache[x] *= transferDer<T,FN_TRANS>(*LnOutTrans, T(0));
							LnOutTrans++;
						}
					}
					
					for(auto kr = uMAX(0); kr < SZ_KER_EDGE; ++kr)
					{
						for(auto x = LINE_BEG; x < LINE_END; ++x)
						{
							for(auto w = uMAX(0); w < SZ_KER_EDGE; ++w)
							{
								*(LnWeightsDlt+w) += *(LnInRow+w) * DerCache[x];
								*(LnGradRow+w) += *(LnWeights+w) *DerCache[x];
							}

							LnInRow++;
							LnGradRow++;
							LnOut++;
						}

						LnInRow += WIDTH_IN - LINE_LEN;
						LnGradRow += WIDTH_IN - LINE_LEN;
						LnWeights += SZ_KER_EDGE;
						LnWeightsDlt += SZ_KER_EDGE;
						LnOut -= LINE_LEN;
					}

					LnWeights -= SZ_KER;
					LnWeightsDlt -= SZ_KER;
				}}
			}

			SX_MC_LAYER_NEXT_FIT;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get error between target and output.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplErr.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplReset.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply optimizations and update parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplApply.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplStore.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplLoad.hpp"
	};
}
