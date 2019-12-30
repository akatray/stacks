// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <stacks/Layer.hpp>

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
	// Scaling direction.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	enum class ScaleDir
	{
		UP,
		DOWN
	};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Scaling operation 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		u64 WIDTH_IN,
		u64 HEIGHT_IN,
		u64 DEPTH_IN,
		ScaleDir DIR,
		FnErr FN_ERR = FnErr::MSE
	>
		
	class Scale2 : public Layer<T>
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto initOW ( void ) { if constexpr(DIR == ScaleDir::UP) return WIDTH_IN*2; else return WIDTH_IN/2; };	
		constexpr static auto WIDTH_OUT = initOW();
		constexpr static auto initOH ( void ) { if constexpr(DIR == ScaleDir::UP) return HEIGHT_IN*2; else return HEIGHT_IN/2; };	
		constexpr static auto HEIGHT_OUT = initOH();
		constexpr static auto SZ_BUF_O = WIDTH_OUT * HEIGHT_OUT * DEPTH_IN;
		constexpr static auto SZ_BUF_I = WIDTH_IN * HEIGHT_IN * DEPTH_IN;
	
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(ALIGNMENT) T OutTrans[SZ_BUF_O];
		alignas(ALIGNMENT) T Gradient[SZ_BUF_I];
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Scale2 ( void ) : OutTrans{}, Gradient{} {}
		~Scale2 ( void ) final {}
		constexpr auto outSz ( void ) const -> u64 final { return SZ_BUF_O; }
		constexpr auto outSzBt ( void ) const -> u64 final { return SZ_BUF_O * sizeof(T); }
		constexpr auto out ( void ) const -> const T* final { return this->OutTrans; }
		constexpr auto gradient ( void ) const -> const T* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute layer.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto exe ( void ) -> void final
		{
			// Upscale input x2.
			if constexpr(DIR == ScaleDir::UP)
			{
				auto ox = u64(0);
				auto oy = u64(0);

				for(auto iy = u64(0); iy < HEIGHT_IN; ++iy) { for(auto ix = u64(0); ix < WIDTH_IN; ++ix)
				{
					for(auto d = u64(0); d < DEPTH_IN; ++d)
					{
						const auto Value = this->Input[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)];

						this->OutTrans[math::index_c(ox, oy, d, WIDTH_OUT, HEIGHT_OUT)] = Value;
						this->OutTrans[math::index_c(ox+1, oy, d, WIDTH_OUT, HEIGHT_OUT)] = Value;
						this->OutTrans[math::index_c(ox, oy+1, d, WIDTH_OUT, HEIGHT_OUT)] = Value;
						this->OutTrans[math::index_c(ox+1, oy+1, d, WIDTH_OUT, HEIGHT_OUT)] = Value;
					}

					ox += 2; if(ox >= WIDTH_OUT) { ox = 0; oy += 2; }
				}}
			}
			

			// Downscale input x2.
			if constexpr(DIR == ScaleDir::DOWN)
			{
				auto ox = u64(0);
				auto oy = u64(0);

				for(auto iy = u64(0); iy < HEIGHT_IN; iy += 2) { for(auto ix = u64(0); ix < WIDTH_IN; ix += 2)
				{
					for(auto d = u64(0); d < DEPTH_IN; ++d)
					{
						auto Sum = T(0.0);

						Sum += this->Input[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)];
						Sum += this->Input[math::index_c(ix+1, iy, d, WIDTH_IN, HEIGHT_IN)];
						Sum += this->Input[math::index_c(ix, iy+1, d, WIDTH_IN, HEIGHT_IN)];
						Sum += this->Input[math::index_c(ix+1, iy+1, d, WIDTH_IN, HEIGHT_IN)];

						this->OutTrans[math::index_c(ox, oy, d, WIDTH_OUT, HEIGHT_OUT)] = Sum;
					}

					++ox; if(ox >= WIDTH_OUT) { ox = 0; ++oy; }
				}}
			}


			// Execute next layer.
			if(this->Front) this->Front->exe();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get error in respect to target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto err ( const T* _Target ) -> T final
		{
			return error<T,FN_ERR>(SZ_BUF_O, _Target, this->OutTrans);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Backpropagate target through network.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const T* _Target, const T _Rate, const T _Error ) -> void final
		{
			// Reset gradient.
			memZero(SZ_BUF_I, this->Gradient);
			
			
			// Upscale.
			if constexpr(DIR == ScaleDir::UP)
			{
				auto ox = u64(0);
				auto oy = u64(0);
				
				for(auto iy = u64(0); iy < HEIGHT_IN; ++iy) { for(auto ix = u64(0); ix < WIDTH_IN; ++ix)
				{
					for(auto d = u64(0); d < DEPTH_IN; ++d)
					{
						u64 o[4];
						o[0] = math::index_c(ox, oy, d, WIDTH_OUT, HEIGHT_OUT);
						o[1] = math::index_c(ox+1, oy, d, WIDTH_OUT, HEIGHT_OUT);
						o[2] = math::index_c(ox, oy+1, d, WIDTH_OUT, HEIGHT_OUT);
						o[3] = math::index_c(ox+1, oy+1, d, WIDTH_OUT, HEIGHT_OUT);


						T DerErr[4];
					
						if(this->Front)
						{
							DerErr[0] = this->Front->gradient()[o[0]];
							DerErr[1] = this->Front->gradient()[o[1]];
							DerErr[2] = this->Front->gradient()[o[2]];
							DerErr[3] = this->Front->gradient()[o[3]];
						}
					
						else
						{
							DerErr[0] = errorDer<T,FN_ERR>(_Target[o[0]], this->OutTrans[o[0]]);
							DerErr[1] = errorDer<T,FN_ERR>(_Target[o[1]], this->OutTrans[o[1]]);
							DerErr[2] = errorDer<T,FN_ERR>(_Target[o[2]], this->OutTrans[o[2]]);
							DerErr[3] = errorDer<T,FN_ERR>(_Target[o[3]], this->OutTrans[o[3]]);
						}


						this->Gradient[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)] = (DerErr[0] + DerErr[1] + DerErr[2] + DerErr[3]);
					}

					ox += 2; if(ox >= WIDTH_OUT) { ox = 0; oy += 2; }
				}}
			}
		

			// Downscale.
			if constexpr(DIR == ScaleDir::DOWN)
			{
				auto ox = u64(0);
				auto oy = u64(0);

				for(auto iy = u64(0); iy < HEIGHT_IN; iy += 2) { for(auto ix = u64(0); ix < WIDTH_IN; ix += 2)
				{
					for(auto d = u64(0); d < DEPTH_IN; ++d)
					{
						const auto o = math::index_c(ox, oy, WIDTH_OUT);
					
						auto DerErr = T(0.0);
						if(this->Front) DerErr = this->Front->gradient()[o];
						else DerErr += errorDer<T,FN_ERR>(_Target[o], this->OutTrans[o]);

						this->Gradient[math::index_c(ix, iy, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						this->Gradient[math::index_c(ix+1, iy, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						this->Gradient[math::index_c(ix, iy+1, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
						this->Gradient[math::index_c(ix+1, iy+1, d, WIDTH_IN, HEIGHT_IN)] = DerErr;
					}

					++ox; if(ox >= WIDTH_OUT) { ox = 0; ++oy; }
				}}
			}


			// Fit backwards.
			if(this->Back) this->Back->fit(nullptr, _Rate, _Error);
		}
	};
}
