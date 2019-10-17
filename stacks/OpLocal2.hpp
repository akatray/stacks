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
// Stacks: Neural Networks Toolkit.
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
	template<u64 W, u64 H, u64 R = 1, Func F = Func::RELU> class OpLocal2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto KER_AREA = u64(((R*2)+1)*((R*2)+1));
		constexpr static auto KER_RDX_MIN = -i64(R);
		constexpr static auto KER_RDX_MAX = i64(R+1);
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(simd::ALIGNMENT) r32 OutTrans[W*H];
		alignas(simd::ALIGNMENT) r32 OutReal[W*H];
		alignas(simd::ALIGNMENT) r32 Gradient[W*H];
		alignas(simd::ALIGNMENT) r32 Weights[W*H*KER_AREA];
		alignas(simd::ALIGNMENT) r32 WeightsDlt[W*H*KER_AREA];

		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpLocal2 ( const r32 _IntMin = 0.001f, const r32 _IntMax = 0.002f )
		{
			if constexpr((W-R*2) < (R*2+1)) static_assert(false, "Kernel does not fit in horizontal space!");
			if constexpr((H-R*2) < (R*2+1)) static_assert(false, "Kernel does not fit in vertical space!");
			
			rng::rbuf(this->Weights, W*H*KER_AREA, _IntMin, _IntMax);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpLocal2 ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr auto outSz ( void ) const -> u64 final { return W*H; }
		constexpr auto outBt ( void ) const -> u64 final { return W*H*sizeof(r32); }
		auto out ( void ) const -> const r32* final { return this->OutTrans; }
		auto gradient ( void ) const -> const r32* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto exe ( const r32* _Input ) -> const r32* final
		{
			const auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;

			
			for(auto y = u64(R); y < u64(H-R); ++y) { for(auto x = u64(R); x < u64(W-R); ++x)
			{
				const auto o = math::index_c(x, y, W);
				auto w = u64(0);
				auto Sum = r32(0.0f);

				for(auto ky = KER_RDX_MIN; ky != KER_RDX_MAX; ++ky) { for(auto kx = KER_RDX_MIN; kx != KER_RDX_MAX; ++kx)
				{
					Sum += this->Input[math::index_c(u64(x+kx), u64(y+ky), W)] * this->Weights[math::index_r(o, w, KER_AREA)];
					++w;
				}}

				if constexpr(F == Func::SIGMOID) this->OutTrans[o] = math::sigmoid(Sum);
				if constexpr(F == Func::TANH) this->OutTrans[o] = math::tanh(Sum);
				if constexpr(F == Func::RELU) this->OutTrans[o] = math::relu(Sum);
				if constexpr(F == Func::PRELU) this->OutTrans[o] = math::prelu(Sum, 0.1f);

				if constexpr((F == Func::RELU) || (F == Func::PRELU)) this->OutReal[o] = Sum;
			}}


			if(this->Back) this->Input = InputCopy;
			if(this->Front) return this->Front->exe(nullptr);
			else return this->out();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto reset ( void ) -> void final
		{
			if(!this->IsLocked) std::memset(this->WeightsDlt, 0, W*H*KER_AREA*sizeof(r32));
			std::memset(this->Gradient, 0, W*H*sizeof(r32));

			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target ) -> void final
		{
			for(auto y = u64(R); y < u64(H-R); ++y) { for(auto x = u64(R); x < u64(W-R); ++x)
			{
				const auto o = math::index_c(x, y, W);
				auto w = u64(0);
				
				auto DerOut = r32(0.0f);
				if(this->Front) DerOut = this->Front->gradient()[o];
				else DerOut = (this->OutTrans[o] - _Target[o]);

				auto DerIn = r32(0.0f);
				if constexpr(F == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::RELU) DerIn = math::reluDer(this->OutReal[o]) * DerOut;
				if constexpr(F == Func::PRELU) DerIn = math::preluDer(this->OutReal[o], 0.1f) * DerOut;

				for(auto ky = KER_RDX_MIN; ky != KER_RDX_MAX; ++ky) { for(auto kx = KER_RDX_MIN; kx != KER_RDX_MAX; ++kx)
				{
					const auto i = math::index_c(u64(x+kx), u64(y+ky), W);
					const auto ow = math::index_r(o, w, KER_AREA);
					
					if(!this->IsLocked) this->WeightsDlt[ow] += (this->Input[i] * DerIn);
					this->Gradient[i] += this->Weights[ow] * DerIn;

					++w;
				}}
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
				simd::mulVecByConstSubFromOut(W*H*KER_AREA, this->Weights, this->WeightsDlt, _Rate);
				
				/*
				if constexpr((F == Func::RELU) || (F == Func::PRELU))
				{
					simd::mulVecByConstSubFromOut(OW*OH*KER_AREA, this->Weights, this->WeightsDlt, _Rate * 0.01f);
				}

				else
				{
					simd::mulVecByConstSubFromOut(OW*OH*KER_AREA, this->Weights, this->WeightsDlt, _Rate);
				}
				*/
			}


			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), W*H*KER_AREA*sizeof(r32));
			
			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), W*H*KER_AREA*sizeof(r32));

			if(this->Front) this->Front->load(_Stream);
		}
	};
}
