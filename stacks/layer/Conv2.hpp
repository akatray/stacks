// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once


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
		bool USE_BIASES = true,
		class FN_TRANS = FnTrRelu<T>,
		FnOptim FN_OPTIM = FnOptim::ADAM
	>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class Conv2 :
		public Layer<T>,
		LDWeights<T, FN_OPTIM, uMAX(((RADIUS*2)+1)*((RADIUS*2)+1))*KERNELS*DEPTH_IN, WIDTH_IN*HEIGHT_IN*DEPTH_IN, WIDTH_IN*HEIGHT_IN*KERNELS, FnInitWeights::DEFAULT>,
		LDBiases<T, WIDTH_IN*HEIGHT_IN*KERNELS, 0, 0, FN_OPTIM>
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

		constexpr static auto SZ_BUF_W = SZ_KER * KERNELS * DEPTH_IN;
		constexpr static auto SZ_BUF_B = WIDTH_IN * HEIGHT_IN * KERNELS;
		constexpr static auto SZ_IN = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_OUT = WIDTH_IN * HEIGHT_IN * KERNELS;
		

		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T OutTemp[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_IN];

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Generated functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_MC_LAYER_TRIVIAL(Conv2, SZ_OUT, this->OutTrans, this->Gradient)


		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute. Optimized for sequential access.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_EXE final
		{
			memZero(SZ_OUT, this->OutTemp);


			for(auto k = uMAX(0); k < KERNELS; ++k)
			{ 
				// Apply kernel on input.
				for(auto d = uMAX(0); d < DEPTH_IN; ++d) { for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y)
				{
					auto LineKernel = this->Weights + math::index_c(0, d, k, SZ_KER, DEPTH_IN);
					auto LineOutTemp = this->OutTemp + math::index_c(RADIUS, y, k, WIDTH_IN, HEIGHT_IN);

					for(auto kr = uMAX(0); kr < SZ_KER_EDGE; ++kr)
					{
						auto LineInput = this->Input + math::index_c(0, y - RADIUS + kr, d, WIDTH_IN, HEIGHT_IN);
						for(auto x = uMAX(0); x < LINE_LEN; ++x) for(auto w = uMAX(0); w < SZ_KER_EDGE; ++w) LineOutTemp[x] += LineInput[x+w] * LineKernel[math::index_c(w, kr, SZ_KER_EDGE)];
					}
				}}
			}

			// Apply biases and transfer values.
			for(auto o = uMAX(0); o < (WIDTH_IN * HEIGHT_IN * KERNELS); ++o)
			{
				if constexpr(USE_BIASES) this->OutTemp[o] += this->Biases[o];
				this->OutTrans[o] = FN_TRANS::trans(this->OutTemp[o]);
			}


			SX_MC_LAYER_NEXT_EXE;
		}


		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate. Optimized for memory order.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_FIT final
		{
			memZero(SZ_IN, this->Gradient);


			if(!this->IsLocked)
			{
				T LineDerTrans[LINE_LEN];
				auto PtrFrontGradient = this->Front->gradient();
				auto PtrTrDerSrc = this->OutTemp;
				if constexpr(!FN_TRANS::RAW) PtrTrDerSrc = this->OutTrans;

				for(auto k = uMAX(0); k < KERNELS; ++k)
				{ 
					// For each channel.
					for(auto d = uMAX(0); d < DEPTH_IN; ++d) { for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y)
					{
						const auto OffKernel = math::index_c(0, d, k, SZ_KER, DEPTH_IN);
						auto LineKernel = this->Weights + OffKernel;
						auto LineKernelDlt = this->WeightsDlt + OffKernel;

						const auto OffOut =  math::index_c(RADIUS, y, k, WIDTH_IN, HEIGHT_IN);
						auto LineOutUn = PtrTrDerSrc + OffOut;
						
						memCopy(LINE_LEN, LineDerTrans, PtrFrontGradient + OffOut);
						for(auto x = 0; x < LINE_LEN; ++x) LineDerTrans[x] *= FN_TRANS::der(LineOutUn[x]);;
						//for(auto x = 0; x < LINE_LEN; ++x) LineDerTrans[x] *= std::clamp<T>(FN_TRANS::der(LineOutUn[x]), -1, 1);;


						for(auto kr = uMAX(0); kr < SZ_KER_EDGE; ++kr)
						{ 
							const auto OffIn = math::index_c(0, y - RADIUS + kr, d, WIDTH_IN, HEIGHT_IN);
							auto LineInput = this->Input + OffIn;
							auto LineGrad = this->Gradient + OffIn;

							for(auto x = 0; x < LINE_LEN; ++x)
							{
								for(auto w = uMAX(0); w < SZ_KER_EDGE; ++w)
								{
									const auto IdxKer = math::index_c(w, kr, SZ_KER_EDGE);
									LineKernelDlt[IdxKer] += LineInput[x+w] * LineDerTrans[x];
									LineGrad[x+w] += LineKernel[IdxKer] * LineDerTrans[x];
								}
							}
						}
					}}
				}

				// Apply biases and transfer values.
				if constexpr(USE_BIASES)
				{
					auto OutNeeded = this->OutTrans;
					if constexpr(FN_TRANS::RAW) OutNeeded = this->OutTemp;

					for(auto o = uMAX(0); o < (WIDTH_IN * HEIGHT_IN * KERNELS); ++o)
					{
						const auto DerErr = this->Front->gradient()[o];
						const auto DerTrans = FN_TRANS::der(OutNeeded[o]) * DerErr;
						this->BiasesDlt[o] += DerTrans;
					}
				}
			}

			else
			{
				T LineDerTrans[LINE_LEN];
				auto PtrFrontGradient = this->Front->gradient();
				auto PtrTrDerSrc = this->OutTemp;
				if constexpr(!FN_TRANS::RAW) PtrTrDerSrc = this->OutTrans;

				for(auto k = uMAX(0); k < KERNELS; ++k)
				{ 
					// For each channel.
					for(auto d = uMAX(0); d < DEPTH_IN; ++d) { for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y)
					{
						const auto OffKernel = math::index_c(0, d, k, SZ_KER, DEPTH_IN);
						auto LineKernel = this->Weights + OffKernel;
						auto LineKernelDlt = this->WeightsDlt + OffKernel;

						const auto OffOut =  math::index_c(RADIUS, y, k, WIDTH_IN, HEIGHT_IN);
						auto LineOutUn = PtrTrDerSrc + OffOut;
						
						memCopy(LINE_LEN, LineDerTrans, PtrFrontGradient + OffOut);
						for(auto x = 0; x < LINE_LEN; ++x) LineDerTrans[x] *= FN_TRANS::der(LineOutUn[x]);;
						//for(auto x = 0; x < LINE_LEN; ++x) LineDerTrans[x] *= std::clamp<T>(FN_TRANS::der(LineOutUn[x]), -1, 1);;


						for(auto kr = uMAX(0); kr < SZ_KER_EDGE; ++kr)
						{ 
							const auto OffIn = math::index_c(0, y - RADIUS + kr, d, WIDTH_IN, HEIGHT_IN);
							auto LineInput = this->Input + OffIn;
							auto LineGrad = this->Gradient + OffIn;

							for(auto x = 0; x < LINE_LEN; ++x)
							{
								for(auto w = uMAX(0); w < SZ_KER_EDGE; ++w)
								{
									const auto IdxKer = math::index_c(w, kr, SZ_KER_EDGE);
									//LineKernelDlt[IdxKer] += LineInput[x+w] * LineDerTrans[x];
									LineGrad[x+w] += LineKernel[IdxKer] * LineDerTrans[x];
								}
							}
						}
					}}
				}
			}

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
