// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <stacks/Op.hpp>
#include <fx/Simd.hpp>
#include <fx/Rng.hpp>

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stacks - Neural Networks Toolkit.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Local dense operation 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<u64 IW, u64 IH, u64 OW, u64 OH, u64 R = 1, Func F = Func::RELU> class OpLocal2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto KER_AREA = u64(((R*2)+1)*((R*2)+1));
		constexpr static auto KER_RDX_MIN = -i64(R);
		constexpr static auto KER_RDX_MAX = i64(R+1);
		constexpr static auto IN_STEP_X = r32(IW-R*2)/OW;
		constexpr static auto IN_STEP_Y = r32(IH-R*2)/OH;
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(simd::ALIGNMENT) r32 OutTrans[OW*OH];
		alignas(simd::ALIGNMENT) r32 OutReal[OW*OH];
		alignas(simd::ALIGNMENT) r32 Gradient[IW*IH];
		alignas(simd::ALIGNMENT) r32 Weights[OW*OH*KER_AREA];
		alignas(simd::ALIGNMENT) r32 WeightsDlt[OW*OH*KER_AREA];

		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpLocal2 ( r32 _IntMin = 0.001f, r32 _IntMax = 0.002f )
		{
			if constexpr((IW-R*2) < (R*2+1)) static_assert(false, "Kernel does not fit in horizontal input space!");
			if constexpr((IH-R*2) < (R*2+1)) static_assert(false, "Kernel does not fit in vertical input space!");
			
			rng::rbuf(this->Weights, OW * OH * KER_AREA, _IntMin, _IntMax);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpLocal2 ( void ) final
		{
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto id ( void ) const -> u64 final { return 2000; }
		auto shpin ( void ) const -> utl::Shape final { return utl::Shape(IW, IH); }
		auto shpout ( void ) const -> utl::Shape final { return utl::Shape(OW, OH); }
		auto output ( void ) -> r32* final { return this->OutTrans; }
		auto gradient ( void ) const -> const r32* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			

			auto ox = u64(0);
			auto oy = u64(0);
			
			for(auto iy = r32(R); iy < r32(IH - R); iy += IN_STEP_Y) { for(auto ix = r32(R); ix < r32(IW - R); ix += IN_STEP_X)
			{
				const auto o = math::index_c(ox, oy, OW);
				
				auto w = u64(0);

				auto Sum = r32(0.0f);

				for(auto ky = KER_RDX_MIN; ky != KER_RDX_MAX; ++ky) { for(auto kx = KER_RDX_MIN; kx != KER_RDX_MAX; ++kx)
				{
					Sum += this->Input[math::index_c(u64(ix+kx), u64(iy+ky), IW)] * this->Weights[math::index_r(o, w, KER_AREA)];
					++w;
				}}

				if constexpr(F == Func::SIGMOID) this->OutTrans[o] = math::sigmoid(Sum);
				if constexpr(F == Func::TANH) this->OutTrans[o] = math::tanh(Sum);
				if constexpr(F == Func::RELU) this->OutTrans[o] = math::relu(Sum);
				if constexpr(F == Func::PRELU) this->OutTrans[o] = math::prelu(Sum, 0.05f);

				if constexpr((F == Func::RELU) || (F == Func::PRELU)) this->OutReal[o] = Sum;

				++ox; if(ox >= OW) { ox = 0; ++oy; }
			}}


			if(this->Back) this->Input = InputCopy;
			if(this->Front) return this->Front->execute(nullptr);
			else return this->output();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto reset ( void ) -> void final
		{
			if(!this->IsLocked) std::memset(this->WeightsDlt, 0, OW*OH*KER_AREA*sizeof(r32));
			std::memset(this->Gradient, 0, IW*IH*sizeof(r32));

			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target ) -> void final
		{
			auto ox = u64(0);
			auto oy = u64(0);
			
			for(auto iy = r32(R); iy < r32(IH-R); iy += IN_STEP_Y) { for(auto ix = r32(R); ix < r32(IW-R); ix += IN_STEP_X)
			{
				const auto o = math::index_c(ox, oy, OW);
				auto w = u64(0);
					
				auto DerOut = r32(0.0f);
				if(this->Front) DerOut = this->Front->gradient()[o];
				else DerOut = (this->OutTrans[o] - _Target[o]);

				auto DerIn = r32(0.0f);
				if constexpr(F == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::RELU) DerIn = math::reluDer(this->OutReal[o]) * DerOut;
				if constexpr(F == Func::PRELU) DerIn = math::preluDer(this->OutReal[o], 0.05f) * DerOut;

				for(auto ky = KER_RDX_MIN; ky != KER_RDX_MAX; ++ky) { for(auto kx = KER_RDX_MIN; kx != KER_RDX_MAX; ++kx)
				{
					const auto i = math::index_c(u64(ix+kx), u64(iy+ky), IW);
					const auto ow = math::index_r(o, w, KER_AREA);
					
					if(!this->IsLocked) this->WeightsDlt[ow] += (this->Input[i] * DerIn);
					this->Gradient[i] += this->Weights[ow] * DerIn;

					++w;
				}}

				++ox; if(ox >= OW) { ox = 0; ++oy; }
			}}


			if(this->Back) this->Back->fit(nullptr);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r32 _Rate = 0.01f ) -> void final
		{
			if(!this->IsLocked)
			{
				if constexpr((F == Func::RELU) || (F == Func::PRELU))
				{
					simd::mulVecByConstSubFromOut(OW * OH * KER_AREA, this->Weights, this->WeightsDlt, _Rate * 0.1f);
				}

				else
				{
					simd::mulVecByConstSubFromOut(OW * OH * KER_AREA, this->Weights, this->WeightsDlt, _Rate);
				}

			}


			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), OW * OH * KER_AREA * sizeof(r32));
			
			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), OW * OH * KER_AREA * sizeof(r32));

			if(this->Front) this->Front->load(_Stream);
		}
	};
}
