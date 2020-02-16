// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include "./../Layer.hpp"
#include "./Dense.hpp"

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
	// Layer for variational autoencoder.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		uMAX SZ_IN,
		uMAX SZ_OUT,
		uMAX PRIORITY = 1,
		FnOptim FN_OPTIM = FnOptim::MOMENTUM,
		FnErr FN_ERR = FnErr::MSE
	>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class Variation :
		public Layer<T>,
		LDOutputs<T, SZ_OUT, SZ_IN * 2, false>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto GRADIENT_SUPPRESS = T(1) / PRIORITY;

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Dense<T, SZ_IN, SZ_OUT * 2, FnTrans::LINEAR, sx::FnInitWeights::NRM_TANH, FN_OPTIM, FN_ERR> MeanDev;
		alignas(ALIGNMENT) T NormalSample[SZ_OUT];

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Generated functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_MC_LAYER_TRIVIAL(Variation, SZ_OUT, this->OutTrans, this->Gradient)

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Variation ( void ) : MeanDev(), NormalSample{} {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_EXE final
		{
			if(this->Back)
			{
				this->MeanDev.setBack(this->Back);
				this->MeanDev.setFront(nullptr);
				this->MeanDev.exe(false);
			}

			else
			{
				this->MeanDev.setInput(this->Input);
				this->MeanDev.exe(false);
			}

			this->setBack(this->Back);


			rng::rbuf_nrm(SZ_OUT, NormalSample, T(0), T(1));
			for(auto o = uMAX(0); o < SZ_OUT; ++o) this->OutTrans[o] = this->MeanDev.out()[o] + this->MeanDev.out()[o + SZ_OUT] * NormalSample[o];


			SX_MC_LAYER_NEXT_EXE;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_FIT final
		{
			for(auto o = uMAX(0); o < SZ_OUT; ++o)
			{
				auto et = this->err(nullptr);
				auto e = math::sqr(this->MeanDev.out()[o]) + std::exp(this->MeanDev.out()[o + SZ_OUT]) - this->MeanDev.out()[o + SZ_OUT] - T(1);

				auto DerErrMean = T(0);
				auto DerErrDev = T(0);

				if(e > T(0.0))
				{
					DerErrMean = (this->MeanDev.out()[o] * T(2)) ;
					DerErrDev = (std::exp(this->MeanDev.out()[o + SZ_OUT]) - T(1)) ;
				}

				const auto DerErrMeanRec = this->Front->gradient()[o];
				const auto DerErrDevRec = this->Front->gradient()[o] * this->NormalSample[o];

				//const auto DerErrMeanRec = this->Front->gradient()[o] * std::lerp(T(1), T(0), std::clamp(et*5, T(0), T(0.9)));
				//const auto DerErrDevRec = this->Front->gradient()[o] * this->NormalSample[o] * std::lerp(T(1), T(0), std::clamp(et*5, T(0), T(0.9)));

				this->Gradient[o] = DerErrMeanRec + DerErrMean;
				this->Gradient[o+SZ_OUT] = DerErrDevRec + DerErrDev;
			}


			if(this->Back)
			{
				this->MeanDev.setFront(this);
				this->MeanDev.setBack(this->Back);
				this->Back->setFront(&this->MeanDev);
			}

			else
			{
				this->MeanDev.setFront(this);
				this->MeanDev.setInput(this->Input);
			}

			this->MeanDev.fit(nullptr, _ErrParam, true);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get KL loss.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_ERR final
		{
			auto Dkl = T(0);

			for(auto o = uMAX(0); o < SZ_OUT; ++o)
			{
				Dkl +=  math::sqr(this->MeanDev.out()[o]) + std::exp(this->MeanDev.out()[o + SZ_OUT]) - this->MeanDev.out()[o + SZ_OUT] - T(1);
			}

			return Dkl * T(0.5);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset delta parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_RESET final
		{
			if(!this->IsLocked)
			{
				this->MeanDev.reset(false);
			}
			
			SX_MC_LAYER_NEXT_RESET;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply optimizations and update parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_APPLY final
		{
			if(!this->IsLocked)
			{
				this->MeanDev.apply(_Rate, _Iter, false);
			}

			SX_MC_LAYER_NEXT_APPLY;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store parameters to stream.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_STORE final
		{
			this->MeanDev.store(_Stream, false);
			SX_MC_LAYER_NEXT_STORE;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load parameters from stream.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_LOAD final
		{
			this->MeanDev.load(_Stream, false);
			SX_MC_LAYER_NEXT_LOAD;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Multi threading utility.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_EXCHANGE final
		{
			auto Master = static_cast<decltype(this)>(_Master);

			this->MeanDev.exchange(&Master->MeanDev, false);

			if(this->Front && _Chain) this->Front->exchange(Master->Front);
		}

	};
}
