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
	// Locally connected layer 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		u64 WIDTH_IN,
		u64 HEIGHT_IN,
		u64 DEPTH_IN,
		u64 RADIUS = 1,
		FnTrans FN_TRANS = FnTrans::PRELU,
		FnOptim FN_OPTIM = FnOptim::ADAM,
		FnErr FN_ERR = FnErr::MSE
	>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class Local2 : public Layer<T>
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto SZ_KER = u64(((RADIUS*2)+1)*((RADIUS*2)+1));
		constexpr static auto SZ_KER_RDX_MIN = -i64(RADIUS);
		constexpr static auto SZ_KER_RDX_MAX = i64(RADIUS+1);
		constexpr static auto SZ_BUF_O = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_BUF_I = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_BUF_W = WIDTH_IN * HEIGHT_IN * DEPTH_IN * SZ_KER;
		constexpr static auto SZ_BUF_B = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		
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

		alignas(ALIGNMENT) T Biases[SZ_BUF_B];
		alignas(ALIGNMENT) T BiasesDlt[SZ_BUF_B];
		alignas(ALIGNMENT) T BiasesM[SZ_BUF_B];
		alignas(ALIGNMENT) T BiasesV[SZ_BUF_B];

		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~Local2 ( void ) final {}
		constexpr auto outSz ( void ) const -> u64 final { return SZ_BUF_O; }
		constexpr auto outSzBt ( void ) const -> u64 final { return SZ_BUF_O * sizeof(T); }
		constexpr auto out ( void ) const -> const T* final { return this->OutTrans; }
		constexpr auto gradient ( void ) const -> const T* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Local2 ( const T _InitMin = T(0.0001), const T _InitMax = T(0.0001) ) :
		OutTrans{}, OutRaw{}, Gradient{},   Weights{}, WeightsDlt{}, WeightsM{}, WeightsV{},   Biases{}, BiasesDlt{}, BiasesM{}, BiasesV{}
		{
			if constexpr((WIDTH_IN-RADIUS * 2) < (RADIUS * 2 + 1)) static_assert(false, "Kernel does not fit in horizontal input space!");
			if constexpr((HEIGHT_IN-RADIUS * 2) < (RADIUS * 2 + 1)) static_assert(false, "Kernel does not fit in vertical input space!");
			
			rng::rbuf(SZ_BUF_W, this->Weights, _InitMin, _InitMax);
			rng::rbuf(SZ_BUF_B, this->Biases, _InitMin, _InitMax);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute layer.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_EXE final
		{
			for(auto d = u64(0); d < DEPTH_IN; ++d)
			{
				for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y) { for(auto x = RADIUS; x < (WIDTH_IN - RADIUS); ++x)
				{
					auto w = u64(0);
					auto Sum = T(0.0);
					for(auto ky = SZ_KER_RDX_MIN; ky != SZ_KER_RDX_MAX; ++ky) { for(auto kx = SZ_KER_RDX_MIN; kx != SZ_KER_RDX_MAX; ++kx)
					{
						Sum += this->Input[math::index_c(x+kx, y+ky, d, WIDTH_IN, HEIGHT_IN)] * this->Weights[math::index_c(w, x, y, d, SZ_KER, WIDTH_IN, HEIGHT_IN)];
						++w;
					}}

					const auto o = math::index_c(x, y, d, WIDTH_IN, HEIGHT_IN);
					this->OutRaw[o] = Sum + this->Biases[o];
				}}
			}


			for(auto o = u64(0); o < SZ_BUF_O; ++o) this->OutTrans[o] = transfer<T,FN_TRANS>(this->OutRaw[o]);
	

			SX_MC_LAYER_NEXT_EXE;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_RESET final
		{
			if(!this->IsLocked)
			{
				memZero(SZ_BUF_W, this->WeightsDlt);
				memZero(SZ_BUF_B, this->BiasesDlt);
			}

			SX_MC_LAYER_NEXT_RESET;
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get error in respect to target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_ERR final
		{
			return error<T,FN_ERR>(SZ_BUF_O, _Target, this->OutTrans);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_FIT final
		{
			memZero(SZ_BUF_I, this->Gradient);


			for(auto d = u64(0); d < DEPTH_IN; ++d)
			{
				for(auto y = RADIUS; y < (HEIGHT_IN - RADIUS); ++y) { for(auto x = RADIUS; x < (WIDTH_IN - RADIUS); ++x)
				{
					const auto o = math::index_c(x, y, d, WIDTH_IN, HEIGHT_IN);
					
					auto DerErr = T(0.0);
					if(this->Front) DerErr = this->Front->gradient()[o];
					else DerErr += errorDer<T,FN_ERR>(_Target[o], this->OutTrans[o]);

					auto DerTrans = transferDer<T,FN_TRANS>(this->OutTrans[o], this->OutRaw[o]) * DerErr;

					auto w = u64(0);
					for(auto ky = SZ_KER_RDX_MIN; ky != SZ_KER_RDX_MAX; ++ky) { for(auto kx = SZ_KER_RDX_MIN; kx != SZ_KER_RDX_MAX; ++kx)
					{
						const auto i = math::index_c(x+kx, y+ky, d, WIDTH_IN, HEIGHT_IN);
						const auto ow = math::index_c(w, x, y, d, SZ_KER, WIDTH_IN, HEIGHT_IN);
							
						if(!this->IsLocked) this->WeightsDlt[ow] += (this->Input[i] * DerTrans);
							
						this->Gradient[i] += (this->Weights[ow] * DerTrans);

						++w;
					}}

					if(!this->IsLocked) this->BiasesDlt[o] += DerTrans;
				}}
			}


			SX_MC_LAYER_NEXT_FIT;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply optimizations and update parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_APPLY final
		{
			if(!this->IsLocked)
			{
				optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_W, this->Weights, this->WeightsDlt, this->WeightsM, this->WeightsV);
				optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_B, this->Biases, this->BiasesDlt, this->BiasesM, this->BiasesV);
			}

			SX_MC_LAYER_NEXT_APPLY;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_STORE final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), SZ_BUF_W * sizeof(T));
			_Stream.write(reinterpret_cast<const char*>(this->Biases), SZ_BUF_B * sizeof(T));
			
			SX_MC_LAYER_NEXT_STORE;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load parameters.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_LOAD final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), SZ_BUF_W * sizeof(T));
			_Stream.read(reinterpret_cast<char*>(this->Biases), SZ_BUF_B * sizeof(T));

			SX_MC_LAYER_NEXT_LOAD;
		}
	};
}
