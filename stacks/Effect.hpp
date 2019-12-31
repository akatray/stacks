// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <stacks/Layer.hpp>

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stacks: Neural Networks Experiment.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Normalize input to 0-1 output.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, u64 SZ_IN, bool ZEROMEAN = false, bool STDDEV = false, bool NORMALIZE = false> class Effect : public Layer<T>
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(ALIGNMENT) T OutTrans[SZ_IN];
		alignas(ALIGNMENT) T Gradient[SZ_IN];
		alignas(ALIGNMENT) T DerStdDev;
		alignas(ALIGNMENT) T DerNorm;

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~Effect ( void ) final {}
		constexpr auto outSz ( void ) const -> u64 final { return SZ_IN; }
		constexpr auto outSzBt ( void ) const -> u64 final { return SZ_IN * sizeof(T); }
		constexpr auto out ( void ) const -> const T* final { return this->OutTrans; }
		constexpr auto gradient ( void ) const -> const T* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto exe ( void ) -> void final
		{
			memCopy(SZ_IN, this->OutTrans, this->Input);

			if constexpr(ZEROMEAN)
			{
				const auto Mean = math::mean(SZ_IN, this->OutTrans);
				vops::subConstFromOut(SZ_IN, this->OutTrans, Mean);
			}

			if constexpr(STDDEV)
			{
				const auto StdDev = math::stddev(SZ_IN, this->OutTrans);
				vops::divOutByConst(SZ_IN, this->OutTrans, StdDev);
				this->DerStdDev = T(1.0) / StdDev;
			}
			
			if constexpr(NORMALIZE)
			{
				const auto MinMax = std::minmax_element(this->OutTrans, this->OutTrans + SZ_IN);
				for(auto o = u64(0); o < SZ_IN; ++o) this->OutTrans[o] = (this->OutTrans[o] - *MinMax.first) / (*MinMax.second - *MinMax.first);
				this->DerNorm = T(1.0) / (*MinMax.second - *MinMax.first);
			}




			// Execute next layer.
			if(this->Front) this->Front->exe();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get error in respect to target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto err ( const T* _Target ) -> T final
		{
			//if constexpr(LATENT)
			//{
				return std::accumulate(this->OutTrans, this->OutTrans + SZ_IN, T(0.0)) / SZ_IN;
			//}

			//else
			//{
			//	return T(0);
			//}
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate target through stack.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const T* _Target, const T _Rate, const T _Error ) -> void final
		{
			auto Err = T(0.0);
			for(auto o = u64(0); o < SZ_IN; ++o) Err += this->OutTrans[o] * o;
			//return Err;
			
			//Update gradient.
			for(auto o = u64(0); o < SZ_IN; ++o)
			{
				auto DerErr = this->Front->gradient()[o];
				if constexpr(STDDEV) DerErr *= this->DerStdDev;
				if constexpr(NORMALIZE) DerErr *= this->DerNorm;
				
				this->Gradient[o] = DerErr;
				
				
				//auto DerIn = this->Front->gradient()[o] * this->DerTrans;
				//auto DerZ = this->OutTrans[o];

				//auto DerO = (DerIn * (this->OutTrans[o] + 0.0)) + ((1.0 - this->OutTrans[o]) * DerZ);
				
				
				//this->Gradient[o] = std::lerp(DerZ*0.9 + DerIn*0.1, DerIn, this->OutTrans[o]);
			}

			// Fit backwards.
			if(this->Back) this->Back->fit(nullptr, _Rate, _Error);
		}
	};
}
