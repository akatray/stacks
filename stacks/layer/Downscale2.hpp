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
	// Buffer for MIN/MAX rerouting.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<uMAX SZ_OUT> struct Downscale2Route
	{
		alignas(ALIGNMENT) uMAX Route[SZ_OUT];
		Downscale2Route ( void ) : Route{}{}
	};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Pooling options.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	enum class FnPool
	{
		AVG,
		ADD,
		MIN,
		MAX
	};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Downscale layer 2d.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		uMAX WIDTH_IN,
		uMAX HEIGHT_IN,
		uMAX DEPTH_IN,
		FnPool FN_POOL = FnPool::MAX
	>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class Downscale2 :
		public Layer<T>,
		Outputs_ld<T, (WIDTH_IN / 2) * (HEIGHT_IN / 2) * DEPTH_IN, WIDTH_IN * HEIGHT_IN * DEPTH_IN>,
		std::conditional_t<(FN_POOL == FnPool::MIN) || (FN_POOL == FnPool::MAX), Downscale2Route<(WIDTH_IN / 2) * (HEIGHT_IN / 2) * DEPTH_IN>, None1>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto WIDTH_OUT = WIDTH_IN / 2;
		constexpr static auto HEIGHT_OUT = HEIGHT_IN / 2;
		constexpr static auto SZ_IN = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
		constexpr static auto SZ_OUT = WIDTH_OUT * HEIGHT_OUT * DEPTH_IN;
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Generated functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_MC_LAYER_TRIVIAL(Downscale2, SZ_OUT, this->OutTrans, this->Gradient)

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_EXE final
		{
			auto ox = uMAX(0);
			auto oy = uMAX(0);

			for(auto iy = uMAX(0); iy < HEIGHT_IN; iy += uMAX(2)) { for(auto ix = uMAX(0); ix < WIDTH_IN; ix += uMAX(2))
			{
				for(auto d = uMAX(0); d < DEPTH_IN; ++d)
				{
					if constexpr((FN_POOL == FnPool::AVG) || (FN_POOL == FnPool::ADD))
					{
						auto Sum = T(0);
						Sum += this->Input[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)];
						Sum += this->Input[math::index_c(ix + uMAX(1), iy, d, WIDTH_IN, HEIGHT_IN)];
						Sum += this->Input[math::index_c(ix, iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)];
						Sum += this->Input[math::index_c(ix + uMAX(1), iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)];

						if constexpr(FN_POOL == FnPool::AVG) this->OutTrans[math::index_c(ox, oy, d, WIDTH_OUT, HEIGHT_OUT)] = Sum * T(0.25);
						if constexpr(FN_POOL == FnPool::ADD) this->OutTrans[math::index_c(ox, oy, d, WIDTH_OUT, HEIGHT_OUT)] = Sum;
					}

					if constexpr((FN_POOL == FnPool::MIN) || (FN_POOL == FnPool::MAX))
					{
						T Candidates[4];
						Candidates[0] = this->Input[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)];
						Candidates[1] = this->Input[math::index_c(ix + uMAX(1), iy, d, WIDTH_IN, HEIGHT_IN)];
						Candidates[2] = this->Input[math::index_c(ix, iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)];
						Candidates[3] = this->Input[math::index_c(ix + uMAX(1), iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)];

						T* Picked;
						if constexpr(FN_POOL == FnPool::MIN) Picked = std::min_element(Candidates, Candidates + 4);
						if constexpr(FN_POOL == FnPool::MAX) Picked = std::max_element(Candidates, Candidates + 4);
					
						const auto o = math::index_c(ox, oy, d, WIDTH_OUT, HEIGHT_OUT);
						this->Route[o] = std::distance(Candidates, Picked);
						this->OutTrans[o] = *Picked;
					}
				}

				++ox; if(ox >= WIDTH_OUT) { ox = uMAX(0); ++oy; }
			}}

			SX_MC_LAYER_NEXT_EXE;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		SX_FNSIG_LAYER_FIT final
		{
			if constexpr((FN_POOL == FnPool::MIN) || (FN_POOL == FnPool::MAX)) memZero(SZ_IN, this->Gradient);
			
			auto ox = uMAX(0);
			auto oy = uMAX(0);

			for(auto iy = uMAX(0); iy < HEIGHT_IN; iy += uMAX(2)) { for(auto ix = uMAX(0); ix < WIDTH_IN; ix += uMAX(2))
			{
				for(auto d = uMAX(0); d < DEPTH_IN; ++d)
				{
					const auto o = math::index_c(ox, oy, d, WIDTH_OUT, HEIGHT_OUT);
					
					if constexpr((FN_POOL == FnPool::AVG) || (FN_POOL == FnPool::ADD))
					{
						auto DerErr = this->Front->gradient()[o];
						if constexpr(FN_POOL == FnPool::AVG) DerErr *= T(0.25);

						this->Gradient[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						this->Gradient[math::index_c(ix + uMAX(1), iy, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						this->Gradient[math::index_c(ix, iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						this->Gradient[math::index_c(ix + uMAX(1), iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)] = DerErr;
					}

					if constexpr((FN_POOL == FnPool::MIN) || (FN_POOL == FnPool::MAX))
					{
						auto DerErr = this->Front->gradient()[o];

						if(this->Route[o] == 0) this->Gradient[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						else if(this->Route[o] == 1) this->Gradient[math::index_c(ix + uMAX(1), iy, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						else if(this->Route[o] == 2) this->Gradient[math::index_c(ix, iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						else this->Gradient[math::index_c(ix + uMAX(1), iy + uMAX(1), d, WIDTH_IN, HEIGHT_IN)] = DerErr;
					}
				}

				++ox; if(ox >= WIDTH_OUT) { ox = uMAX(0); ++oy; }
			}}

			SX_MC_LAYER_NEXT_FIT;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get output error in respect to argument.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplErr.hpp"

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Multi threading utility.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#include "./data/ComImplExchangeSkip.hpp"
	};
}
