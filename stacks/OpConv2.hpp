/*
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
	// Convoliutional operation 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<u64 IW, u64 IH, u64 ID, u64 OW, u64 OH, u64 OD, u64 K = 1, u64 R = 1, Func F = Func::RELU> class OpConv2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Compile time constants.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		constexpr static auto KER_AREA = u64(((R*2)+1)*((R*2)+1));
		constexpr static auto KER_RDX_MIN = -i64(R);
		constexpr static auto KER_RDX_MAX = i64(R+1);
		constexpr static auto IN_STEP_X = r32(IW-R*2)/(OW-R*2);
		constexpr static auto IN_STEP_Y = r32(IH-R*2)/(OH-R*2);
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		alignas(simd::ALIGNMENT) r32 OutTrans[OW*OH*OD];
		alignas(simd::ALIGNMENT) r32 OutReal[OW*OH*OD];
		alignas(simd::ALIGNMENT) r32 Gradient[IW*IH*ID];
		alignas(simd::ALIGNMENT) r32 Weights[K*KER_AREA];
		alignas(simd::ALIGNMENT) r32 WeightsDlt[K*KER_AREA];

		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpConv2 ( const r32 _InitMag = 0.1f )
		{
			if constexpr((IW-R*2) < (R*2+1)) static_assert(false, "Kernel does not fit in horizontal input space!");
			if constexpr((IH-R*2) < (R*2+1)) static_assert(false, "Kernel does not fit in vertical input space!");
			
			rng::rbuf(this->Weights, K*KER_AREA, -_InitMag, _InitMag);
			auto Sum = math::sum(K*KER_AREA, this->Weights);
			for(auto k = u64(0); k < K*KER_AREA; ++k) this->Weights[k] /= Sum;
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpConv2 ( void ) final
		{
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto id ( void ) const -> u64 final { return 3000; }
		auto shpin ( void ) const -> utl::Shape final { return utl::Shape(IW, IH, ID); }
		auto shpout ( void ) const -> utl::Shape final { return utl::Shape(OW, OH, OD); }
		auto output ( void ) -> r32* final { return this->OutTrans; }
		auto gradient ( void ) const -> const r32* final { return this->Gradient; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			

			std::memset(this->OutTrans, 0, OW*OH*OD*sizeof(r32));
			std::memset(this->OutReal, 0, OW*OH*OD*sizeof(r32));

			for(auto k = u64(0); k < K; ++k)
			{
				for(auto id = u64(0); id < ID; ++id) { for(auto y = u64(R); y < u64(IH-R); ++y) { for(auto x = u64(R); x < u64(IW-R); ++x)
				{
					const auto o = math::index_c(x, y, k, OW, OH);
					auto w = u64(0);
					auto Sum = r32(0.0f);

					for(auto ky = KER_RDX_MIN; ky != KER_RDX_MAX; ++ky) { for(auto kx = KER_RDX_MIN; kx != KER_RDX_MAX; ++kx)
					{
						Sum += this->Input[math::index_c(x+kx, y+ky, id, IW, IH)] * this->Weights[math::index_c(w, k, KER_AREA)];
						++w;
					}}

					if constexpr(F == Func::SIGMOID) this->OutTrans[o] += math::sigmoid(Sum);
					if constexpr(F == Func::TANH) this->OutTrans[o] += math::tanh(Sum);
					if constexpr(F == Func::RELU) this->OutTrans[o] += math::relu(Sum);
					if constexpr(F == Func::PRELU) this->OutTrans[o] += math::prelu(Sum, 0.5f);

					if constexpr((F == Func::RELU) || (F == Func::PRELU)) this->OutReal[o] += Sum;
				}}}
			}


			if(this->Back) this->Input = InputCopy;
			if(this->Front) return this->Front->execute(nullptr);
			else return this->output();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto reset ( void ) -> void final
		{
			if(!this->IsLocked) std::memset(this->WeightsDlt, 0, K*KER_AREA*sizeof(r32));
			std::memset(this->Gradient, 0, IW*IH*ID*sizeof(r32));

			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target ) -> void final
		{
			for(auto k = u64(0); k < K; ++k)
			{
				for(auto id = u64(0); id < ID; ++id) { for(auto y = u64(R); y < u64(IH-R); ++y) { for(auto x = u64(R); x < u64(IW-R); ++x)
				{
					const auto o = math::index_c(x, y, k, OW, OH);
					auto w = u64(0);
					
					auto DerOut = r32(0.0f);
					if(this->Front) DerOut = this->Front->gradient()[o];
					else DerOut = (this->OutTrans[o] - _Target[o]);

					auto DerIn = r32(0.0f);
					if constexpr(F == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[o]) * DerOut;
					if constexpr(F == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[o]) * DerOut;
					if constexpr(F == Func::RELU) DerIn = math::reluDer(this->OutReal[o]) * DerOut;
					if constexpr(F == Func::PRELU) DerIn = math::preluDer(this->OutReal[o], 0.5f) * DerOut;

					for(auto ky = KER_RDX_MIN; ky != KER_RDX_MAX; ++ky) { for(auto kx = KER_RDX_MIN; kx != KER_RDX_MAX; ++kx)
					{
						const auto i = math::index_c(x+kx, y+ky, id, IW, IH);
						const auto ow = math::index_c(w, k, KER_AREA);
					
						if(!this->IsLocked) this->WeightsDlt[ow] += (this->Input[i] * DerIn);
						this->Gradient[i] += this->Weights[ow] * DerIn;

						++w;
					}}
				}}}
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
				if constexpr((F == Func::RELU) || (F == Func::PRELU))
				{
					simd::mulVecByConstSubFromOut(K*KER_AREA, this->Weights, this->WeightsDlt, _Rate * 0.001f);
				}

				else
				{
					simd::mulVecByConstSubFromOut(K*KER_AREA, this->Weights, this->WeightsDlt, _Rate);
				}

			}


			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), K*KER_AREA*sizeof(r32));
			
			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load operation's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), K*KER_AREA*sizeof(r32));

			if(this->Front) this->Front->load(_Stream);
		}
	};
}
*/