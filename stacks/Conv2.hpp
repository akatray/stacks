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
	// Convoliutional operation 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		u64 WIDTH_IN,
		u64 HEIGHT_IN,
		u64 DEPTH_IN,
		u64 KERNELS = 1,
		u64 RADIUS = 1,
		FnTrans FN_TRANS = FnTrans::RELU,
		Optim OPTIM = Optim::ADAM,
		FnErr FN_ERR = FnErr::MSE
	>

	class Conv2 : public Layer<T>
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto SZ_KER = u64(((RADIUS*2)+1)*((RADIUS*2)+1));
		constexpr static auto SZ_KER_RDX_MIN = -i64(RADIUS);
		constexpr static auto SZ_KER_RDX_MAX = i64(RADIUS+1);
		constexpr static auto SZ_BUF_O = WIDTH_IN * HEIGHT_IN * KERNELS;
		constexpr static auto SZ_BUF_I = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_BUF_W = SZ_KER * KERNELS * DEPTH_IN;
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(ALIGNMENT) T OutTrans[SZ_BUF_O];
		alignas(ALIGNMENT) T OutRaw[SZ_BUF_O];
		alignas(ALIGNMENT) T Gradient[SZ_BUF_I];
		alignas(ALIGNMENT) T Weights[SZ_BUF_W];
		alignas(ALIGNMENT) T WeightsDlt[SZ_BUF_W];
		alignas(ALIGNMENT) T WeightsM[SZ_BUF_W];
		alignas(ALIGNMENT) T WeightsV[SZ_BUF_W];
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Conv2 ( const T _InitMag = 0.1f) : OutTrans{}, OutRaw{}, Gradient{}, Weights{}, WeightsDlt{}, WeightsM{}, WeightsV{}
		{
			if constexpr((WIDTH_IN-RADIUS * 2) < (RADIUS * 2 + 1)) static_assert(false, "Kernel does not fit in horizontal input space!");
			if constexpr((HEIGHT_IN-RADIUS * 2) < (RADIUS * 2 + 1)) static_assert(false, "Kernel does not fit in vertical input space!");
			
			rng::rbuf(SZ_BUF_W, this->Weights, 0.0001, 0.001);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~Conv2 ( void ) final {}
		constexpr auto outSz ( void ) const -> u64 final { return SZ_BUF_O; }
		constexpr auto outSzBt ( void ) const -> u64 final { return SZ_BUF_O * sizeof(T); }
		constexpr auto out ( void ) const -> const T* final { return this->OutTrans; }
		constexpr auto gradient ( void ) const -> const T* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute layer.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto exe ( void ) -> void final
		{
			// Clear raw buffer.
			memZero(SZ_BUF_O, this->OutRaw);

			// For each kernel.
			for(auto k = u64(0); k < KERNELS; ++k)
			{
				// For each output pixel.
				for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y) { for(auto x = RADIUS; x < (WIDTH_IN - RADIUS); ++x)
				{
					// Precalculate indexes.
					const auto o = math::index_c(x, y, k, WIDTH_IN, HEIGHT_IN);
					auto w = u64(0);
					
					// For each kernel element.
					auto Sum = T(0.0);
					for(auto ky = SZ_KER_RDX_MIN; ky != SZ_KER_RDX_MAX; ++ky) { for(auto kx = SZ_KER_RDX_MIN; kx != SZ_KER_RDX_MAX; ++kx) { for(auto d = u64(0); d < DEPTH_IN; ++d)
					{
						Sum += this->Input[math::index_c(x+kx, y+ky, d, WIDTH_IN, HEIGHT_IN)] * this->Weights[math::index_c(w, k, SZ_KER)];
						++w;
					}}}

					// Store raw value.
					this->OutRaw[o] += Sum;
				}}

				// Apply transfer function.
				for(auto o = u64(0); o < SZ_BUF_O; ++o) this->OutTrans[o] = transfer<T,FN_TRANS>(this->OutRaw[o]);
			}

			// Execute next layer.
			if(this->Front) this->Front->exe();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto reset ( void ) -> void final
		{
			if(!this->IsLocked) memZero(SZ_BUF_W, this->WeightsDlt);
			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get error in respect to target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto err ( const T* _Target ) -> T final
		{
			return error<T,FN_ERR>(SZ_BUF_O, _Target, this->OutTrans);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate target through stack.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const T* _Target, const T _Rate, const T _Error ) -> void final
		{
			// Clear gradient.
			memZero(SZ_BUF_I, this->Gradient);

			auto Red = 1.0;// / WIDTH_IN/ HEIGHT_IN ;
			
			
			// For each kernel.
			for(auto k = u64(0); k < KERNELS; ++k)
			{
				// For each output pixel.
				for(auto y = u64(RADIUS); y < u64(HEIGHT_IN-RADIUS); ++y) { for(auto x = u64(RADIUS); x < u64(WIDTH_IN-RADIUS); ++x)
				{
					// Precalculate indexes.
					const auto o = math::index_c(x, y, k, WIDTH_IN, HEIGHT_IN);
					auto w = u64(0);
					
					// Error derivative.
					auto DerErr = T(0.0);
					if(this->Front) DerErr = this->Front->gradient()[o];
					else DerErr += errorDer<T,FN_ERR>(_Target[o], this->OutTrans[o]);

					// Transfer derivative.
					auto DerTrans = transferDer<T,FN_TRANS>(this->OutTrans[o], this->OutRaw[o]) * DerErr;

					// For each kernel element.
					for(auto ky = SZ_KER_RDX_MIN; ky != SZ_KER_RDX_MAX; ++ky) { for(auto kx = SZ_KER_RDX_MIN; kx != SZ_KER_RDX_MAX; ++kx) { for(auto d = u64(0); d < DEPTH_IN; ++d)
					{
						// Precalculate indexes.
						const auto i = math::index_c(x+kx, y+ky, d, WIDTH_IN, HEIGHT_IN);
						const auto ow = math::index_c(w, k, SZ_KER);
							
						// Update deltas.
						if(!this->IsLocked) this->WeightsDlt[ow] += (this->Input[i] * DerTrans * Red);
							
						// Update gradient.
						this->Gradient[i] += (this->Weights[ow] * DerTrans);

						// Update index.
						++w;
					}}}
				}}
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
				if constexpr(OPTIM == Optim::NONE)
				{
					vops::mulVecByConstSubFromOut(SZ_BUF_W, this->Weights, this->WeightsDlt, _Rate);
				}


				// Momentum.
				if constexpr(OPTIM == Optim::MOMENTUM)
				{
					// For weights.
					for(auto w = u64(0); w < SZ_BUF_W; ++w)
					{
						this->WeightsM[w] = (this->WeightsM[w] * BETA1) + (this->WeightsDlt[w] * BETA1F);
						this->Weights[w] -= _Rate * this->WeightsM[w];
					}
				}


				// Adam.
				if constexpr(OPTIM == Optim::ADAM)
				{
					// For weights.
					for(auto w = u64(0); w < SZ_BUF_W; ++w)
					{
						this->WeightsM[w] = (this->WeightsM[w] * BETA1) + (this->WeightsDlt[w] * BETA1F);
						this->WeightsV[w] = (this->WeightsV[w] * BETA2) + ((this->WeightsDlt[w] * this->WeightsDlt[w]) * BETA2F);
						this->Weights[w] -= _Rate * ((this->WeightsM[w] / BETA1F) / (std::sqrt(this->WeightsV[w] / BETA2F) + EPSILON));
					}
				}
			}


			// Forward to next operation.
			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), SZ_BUF_W * sizeof(T));
			
			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), SZ_BUF_W * sizeof(T));

			if(this->Front) this->Front->load(_Stream);
		}
	};
}
