// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <stacks/Op.hpp>
#include <fx/Simd.hpp>

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stacks: Neural Networks Toolkit.
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
	template<u64 IW, u64 IH, ScaleDir DIR> class OpScale2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto initOW ( void ) { if constexpr(DIR == ScaleDir::UP) return IW*2; else return IW/2; };	
		constexpr static auto OW = initOW();
		constexpr static auto initOH ( void ) { if constexpr(DIR == ScaleDir::UP) return IH*2; else return IH/2; };	
		constexpr static auto OH = initOH();
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(simd::ALIGNMENT) r32 OutTrans[OW*OH];
		alignas(simd::ALIGNMENT) u64 OutIdx[OW*OH];
		alignas(simd::ALIGNMENT) r32 Gradient[IW*IH];
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpScale2 ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr auto outSz ( void ) const -> u64 final { return OW*OH; }
		constexpr auto outBt ( void ) const -> u64 final { return OW*OH*sizeof(r32); }
		auto out ( void ) const -> const r32* final { return this->OutTrans; }
		auto gradient ( void ) const -> const r32* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto exe ( const r32* _Input ) -> const r32* final
		{
			const auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;

			
			if constexpr(DIR == ScaleDir::UP)
			{
				auto ox = u64(0);
				auto oy = u64(0);
				
				for(auto iy = u64(0); iy < u64(IH); ++iy) { for(auto ix = u64(0); ix < u64(IW); ++ix)
				{
					const auto Value = this->Input[math::index_c(ix, iy, IW)];
					
					this->OutTrans[math::index_c(ox, oy, OW)] = Value;
					this->OutTrans[math::index_c(ox+1, oy, OW)] = Value;
					this->OutTrans[math::index_c(ox, oy+1, OW)] = Value;
					this->OutTrans[math::index_c(ox+1, oy+1, OW)] = Value;

					ox += 2; if(ox >= OW) { ox = 0; oy += 2; }
				}}
			}


			if constexpr(DIR == ScaleDir::DOWN)
			{
				std::memset(this->OutTrans, 0, OW*OH*sizeof(r32));
				
				auto ox = u64(0);
				auto oy = u64(0);

				for(auto iy = u64(0); iy < u64(IH); iy += 2) { for(auto ix = u64(0); ix < u64(IW); ix += 2)
				{
					const auto Value0 = this->Input[math::index_c(ix, iy, IW)];
					const auto Value1 = this->Input[math::index_c(ix+1, iy, IW)];
					const auto Value2 = this->Input[math::index_c(ix, iy+1, IW)];
					const auto Value3 = this->Input[math::index_c(ix+1, iy+1, IW)];

					const auto o = math::index_c(ox, oy, OW);

					if(Value0 > this->OutTrans[o]) { this->OutTrans[o] = Value0; this->OutIdx[o] = 0; }
					if(Value1 > this->OutTrans[o]) { this->OutTrans[o] = Value1; this->OutIdx[o] = 1; }
					if(Value2 > this->OutTrans[o]) { this->OutTrans[o] = Value2; this->OutIdx[o] = 2; }
					if(Value3 > this->OutTrans[o]) { this->OutTrans[o] = Value3; this->OutIdx[o] = 3; }

					++ox; if(ox >= OW) { ox = 0; ++oy; }
				}}
			}
			

			if(this->Back) this->Input = InputCopy;
			if(this->Front) return this->Front->exe(nullptr);
			else return this->out();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset gradient.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto reset ( void ) -> void final
		{
			std::memset(this->Gradient, 0, IW*IH*sizeof(r32));
			
			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target ) -> void final
		{
			if constexpr(DIR == ScaleDir::UP)
			{
				auto ix = r32(0.0f);
				auto iy = r32(0.0f);
				
				for(auto oy = u64(0); oy < u64(OH); ++oy) { for(auto ox = u64(0); ox < u64(OW); ++ox)
				{
					const auto o = math::index_c(ox, oy, OW);

					auto DerOut = r32(0.0f);
					if(this->Front) DerOut = this->Front->gradient()[o];
					else DerOut = (this->OutTrans[o] - _Target[o]);

					this->Gradient[math::index_c(ix, iy, IW)] += DerOut;

					ix += 0.5f; if(ix >= IW) { ix = 0.0f; iy += 0.5; }
				}}
			}


			if constexpr(DIR == ScaleDir::DOWN)
			{
				auto ox = u64(0);
				auto oy = u64(0);

				for(auto iy = u64(0); iy < u64(IH); iy += 2) { for(auto ix = u64(0); ix < u64(IW); ix += 2)
				{
					const auto o = math::index_c(ox, oy, OW);
					
					auto DerOut = r32(0.0f);
					if(this->Front) DerOut = this->Front->gradient()[o];
					else DerOut = (this->OutTrans[o] - _Target[o]);

					if(this->OutIdx[o] == 0) this->Gradient[math::index_c(ix, iy, IW)] = DerOut;
					else if(this->OutIdx[o] == 1) this->Gradient[math::index_c(ix+1, iy, IW)] = DerOut;
					else if(this->OutIdx[o] == 2) this->Gradient[math::index_c(ix, iy+1, IW)] = DerOut;
					else this->Gradient[math::index_c(ix+1, iy+1, IW)] = DerOut;


					++ox; if(ox >= OW) { ox = 0; ++oy; }
				}}
			}


			if(this->Back) this->Back->fit(nullptr);
		}
	};
}
