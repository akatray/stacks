// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <stacks/Layer.hpp>
#include <fx/Rng.hpp>

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
	// Dense layer.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		u64 SZ_IN,
		u64 SZ_OUT,
		FnTrans FN_TRANS = FnTrans::SIGMOID,
		FnOptim OPTIM = FnOptim::ADAM,
		FnErr FN_ERR = FnErr::MSE
	>
	
	class Dense : public Layer<T>
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto SZ_BUF_W = SZ_OUT * SZ_IN;
		constexpr static auto SZ_BUF_B = SZ_OUT;
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T OutRaw[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_IN];
		alignas(ALIGNMENT) T Weights[SZ_BUF_W];
		alignas(ALIGNMENT) T WeightsDlt[SZ_BUF_W];
		alignas(ALIGNMENT) T WeightsM[SZ_BUF_W];
		alignas(ALIGNMENT) T WeightsV[SZ_BUF_W];
		alignas(ALIGNMENT) T Biases[SZ_BUF_B];
		alignas(ALIGNMENT) T BiasesDlt[SZ_BUF_B];
		alignas(ALIGNMENT) T BiasesM[SZ_BUF_B];
		alignas(ALIGNMENT) T BiasesV[SZ_BUF_B];
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Dense ( const T _IntMin = 0.001, const T _IntMax = 0.01 ) : OutTrans{}, OutRaw{}, Gradient{}, Weights{}, WeightsDlt{}, WeightsM{}, WeightsV{}, Biases{}, BiasesDlt{}, BiasesM{}, BiasesV{}
		{
			rng::rbuf(SZ_BUF_W, this->Weights, _IntMin, _IntMax);
			rng::rbuf(SZ_BUF_B, this->Biases, _IntMin, _IntMax);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~Dense ( void ) final {}
		constexpr auto outSz ( void ) const -> u64 final { return SZ_OUT; }
		constexpr auto out ( void ) const -> const T* final { return this->OutTrans; }
		constexpr auto gradient ( void ) const -> const T* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute layer.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto exe ( void ) -> void final
		{
			// For each output node.
			for(auto o = u64(0); o < SZ_OUT; ++o)
			{
				this->OutRaw[o] = vops::mulVecByVecSum(SZ_IN, this->Input, &this->Weights[math::index_c(0, o, SZ_IN)]);
				this->OutTrans[o] = transfer<T,FN_TRANS>(this->OutRaw[o] + this->Biases[o]) ;
			}


			// Execute next layer.
			if(this->Front) this->Front->exe();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto reset ( void ) -> void final
		{
			// Reset only if unlocked.
			if(!this->IsLocked)
			{
				memZero(SZ_BUF_W, this->WeightsDlt);
				memZero(SZ_BUF_B, this->BiasesDlt);
			}
			
			
			// Reset next layer.
			if(this->Front) this->Front->reset();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get error in respect to target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto err ( const T* _Target ) -> T final
		{
			return error<T,FN_ERR>(SZ_OUT, _Target, this->OutTrans);
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate target through stack.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const T* _Target, const T _Rate, const T _Error ) -> void final
		{
			// Clear gradient.
			memZero(SZ_IN, this->Gradient);


			// For each node.
			for(auto o = u64(0); o < SZ_OUT; ++o)
			{
				// Error derivitive.
				auto DerErr = T(0.0);
				if(this->Front) DerErr = this->Front->gradient()[o];
				else DerErr += errorDer<T,FN_ERR>(_Target[o], this->OutTrans[o]);

				//DerErr *= _Error;

				// Transfer derivitive.
				auto DerTrans = transferDer<T,FN_TRANS>(this->OutTrans[o], this->OutRaw[o]) * DerErr;

				// Update gradient.
				vops::mulVecByConstAddToOut<T>(SZ_IN, this->Gradient, &this->Weights[math::index_c(0, o, SZ_IN)], DerTrans);

				// Update weights deltas.
				vops::mulVecByConstAddToOut<T>(SZ_IN, &this->WeightsDlt[math::index_c(0, o, SZ_IN)], this->Input, DerTrans);

				// Update biases deltas.
				this->BiasesDlt[o] += DerTrans;
			}


			// Fit backwards.
			if(this->Back) this->Back->fit(nullptr, _Rate, _Error);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply optimizations and update parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r64 _Rate ) -> void final
		{
			// Apply only if unlocked.
			if(!this->IsLocked)
			{
				// No optimizations.
				if constexpr(OPTIM == FnOptim::NONE)
				{
					vops::mulVecByConstSubFromOut(SZ_BUF_W, this->Weights, this->WeightsDlt, _Rate);
				}


				// Momentum.
				if constexpr(OPTIM == FnOptim::MOMENTUM)
				{
					// For weights.
					for(auto w = u64(0); w < SZ_BUF_W; ++w)
					{
						this->WeightsM[w] = (this->WeightsM[w] * BETA1) + (this->WeightsDlt[w] * BETA1F);
						this->Weights[w] -= _Rate * this->WeightsM[w];
					}
					
					// For biases.
					for(auto b = u64(0); b < SZ_BUF_B; ++b)
					{
						this->BiasesM[b] = (this->BiasesM[b] * BETA1) + (this->BiasesDlt[b] * BETA1F);
						this->Biases[b] -= _Rate * this->BiasesM[b];
					}
				}


				// Adam.
				if constexpr(OPTIM == FnOptim::ADAM)
				{
					// For weights.
					for(auto w = u64(0); w < SZ_BUF_W; ++w)
					{
						this->WeightsM[w] = (this->WeightsM[w] * BETA1) + (this->WeightsDlt[w] * BETA1F);
						this->WeightsV[w] = (this->WeightsV[w] * BETA2) + ((this->WeightsDlt[w] * this->WeightsDlt[w]) * BETA2F);

						this->Weights[w] -= _Rate * ((this->WeightsM[w] / BETA1F) / (std::sqrt(this->WeightsV[w] / BETA2F) + EPSILON));
					}

					// For biases.
					for(auto b = u64(0); b < SZ_BUF_B; ++b)
					{
						this->BiasesM[b] = (this->BiasesM[b] * BETA1) + (this->BiasesDlt[b] * BETA1F);
						this->BiasesV[b] = (this->BiasesV[b] * BETA2) + ((this->BiasesDlt[b] * this->BiasesDlt[b]) * BETA2F);

						this->Biases[b] -= _Rate * ((this->BiasesM[b] / BETA1F) / (std::sqrt(this->BiasesV[b] / BETA2F) + EPSILON));
					}
				}
			}


			// Apply next layer.
			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), SZ_BUF_W * sizeof(T));
			_Stream.write(reinterpret_cast<const char*>(this->Biases), SZ_BUF_B * sizeof(T));
			
			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), SZ_BUF_W * sizeof(T));
			_Stream.read(reinterpret_cast<char*>(this->Biases), SZ_BUF_B * sizeof(T));
			
			if(this->Front) this->Front->load(_Stream);
		}
	};
}
