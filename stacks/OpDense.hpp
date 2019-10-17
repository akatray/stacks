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
	// Dense operation.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<u64 IN, u64 OUT, Func F = Func::SIGMOID> class OpDense : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(simd::ALIGNMENT) r32 OutTrans[OUT];
		alignas(simd::ALIGNMENT) r32 OutReal[OUT];
		alignas(simd::ALIGNMENT) r32 Gradient[IN];
		alignas(simd::ALIGNMENT) r32 Weights[OUT*IN];
		alignas(simd::ALIGNMENT) r32 WeightsDlt[OUT*IN];
		alignas(simd::ALIGNMENT) r32 Biases[OUT];
		alignas(simd::ALIGNMENT) r32 BiasesDlt[OUT];

		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpDense ( const r32 _IntMin = -0.001f, const r32 _IntMax = 0.001f ) : OutTrans{}, OutReal{}, Gradient{}, Weights{}, WeightsDlt{}, Biases{}, BiasesDlt{}
		{
			rng::rbuf(this->Weights, OUT * IN, _IntMin, _IntMax);
			rng::rbuf(this->Biases, OUT, _IntMin, _IntMax);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpDense ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr auto outSz ( void ) const -> u64 final { return OUT; }
		constexpr auto outBt ( void ) const -> u64 final { return OUT*sizeof(r32); }
		constexpr auto out ( void ) const -> const r32* final { return this->OutTrans; }
		constexpr auto gradient ( void ) const -> const r32* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto exe ( const r32* _Input ) -> const r32* final
		{
			const auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			

			for(auto o = u64(0); o < OUT; ++o)
			{
				auto Sum = simd::mulVecByVecSum(IN, this->Input, &this->Weights[math::index_c(o, 0, IN)]) + this->Biases[o];
				
				if constexpr(F == Func::SIGMOID) this->OutTrans[o] = math::sigmoid(Sum);
				if constexpr(F == Func::TANH) this->OutTrans[o] = math::tanh(Sum);
				if constexpr(F == Func::RELU) this->OutTrans[o] = math::relu(Sum);
				if constexpr(F == Func::PRELU) this->OutTrans[o] = math::prelu(Sum, 0.1f);

				if constexpr((F == Func::RELU) || (F == Func::PRELU)) this->OutReal[o] = Sum;
			}


			if(this->Back) this->Input = InputCopy;
			if(this->Front) return this->Front->exe(nullptr);
			else return this->out();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto reset ( void ) -> void final
		{
			if(!this->IsLocked)
			{
				std::memset(this->WeightsDlt, 0, OUT * IN * sizeof(r32));
				std::memset(this->BiasesDlt, 0, OUT * sizeof(r32));
			}

			std::memset(this->Gradient, 0, IN * sizeof(r32));


			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target ) -> void final
		{
			for(auto o = u64(0); o < OUT; ++o)
			{
				auto DerOut = r32(0.0f);
				if(this->Front) DerOut = this->Front->gradient()[o];
				else DerOut = this->OutTrans[o] - _Target[o];

				auto DerIn = r32(0.0f);
				if constexpr(F == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::RELU) DerIn = math::reluDer(this->OutReal[o]) * DerOut;
				if constexpr(F == Func::PRELU) DerIn = math::preluDer(this->OutReal[o], 0.1f) * DerOut;

				simd::mulVecByConstAddToOut(IN, this->Gradient, &this->Weights[math::index_c(o, 0, IN)], DerIn);
				if(!this->IsLocked) simd::mulVecByConstAddToOut(IN, &this->WeightsDlt[math::index_c(o, 0, IN)], this->Input, DerIn);
				if(!this->IsLocked) this->BiasesDlt[o] += DerIn;
			}


			if(this->Back) this->Back->fit(nullptr);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r32 _Rate = 0.01f ) -> void final
		{
			if(!this->IsLocked)
			{
				simd::mulVecByConstSubFromOut(OUT * IN, this->Weights, this->WeightsDlt, _Rate);
				simd::mulVecByConstSubFromOut(OUT, this->Biases, this->BiasesDlt, _Rate);
			}


			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), OUT * IN * sizeof(r32));
			_Stream.write(reinterpret_cast<const char*>(this->Biases), OUT * sizeof(r32));
			
			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), OUT * IN * sizeof(r32));
			_Stream.read(reinterpret_cast<char*>(this->Biases), OUT * sizeof(r32));
			
			if(this->Front) this->Front->load(_Stream);
		}
	};
}