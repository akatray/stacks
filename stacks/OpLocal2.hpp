// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <stacks/Op.hpp>
#include <fx/Buffer.hpp>
#include <fx/Rng.hpp>
#include <fx/Simd.hpp>
#include <type_traits>
#include <sstream>

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stacks namespace.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Local dense network 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<Op::Func F, u64 R, u64 IW, u64 IH, u64 OW, u64 OH> class OpLocal2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		u64 KerRadius;
		u64 KerArea;
		Buffer<r32> OutTrans;
		Buffer<r32> OutReal;
		Buffer<r32> Gradient;
		Buffer<Buffer<r32>> Weights;
		Buffer<Buffer<r32>> WeightsDlt;
		
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpLocal2 ( r32 _IntMin = 0.001f, r32 _IntMax = 0.002f ) : Op(utl::Shape(IW, IH), utl::Shape(OW, OH))
		{
			this->KerRadius = R;
			this->KerArea = ((R * 2) + 1) * ((R * 2) + 1);

			this->OutTrans.resize(this->ShpOut.size(), simd::AllocSimd);
			if constexpr((F == Func::RELU) || (F == Func::PRELU)) this->OutReal.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Gradient.resize(this->ShpIn.size(), simd::AllocSimd);

			this->Weights.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) {this->Weights[o].resize(this->KerArea, simd::AllocSimd); rng::rbuf(this->Weights[o](), this->Weights[o].size(), _IntMin, _IntMax);}
			this->WeightsDlt.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsDlt[o].resize(this->KerArea, simd::AllocSimd);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpLocal2 ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto id ( void ) const -> u64 final { return 2000; }
		auto output ( void ) -> r32* final { return this->OutTrans(); }
		auto gradient ( void ) const -> const r32* final { return this->Gradient(); }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			

			constexpr auto RdxMin = -i64(R);
			constexpr auto RdxMax = i64(R + 1);

			constexpr auto init_sox = []() -> auto { if constexpr(IW > OW) return r32(OW) / IW; else return r32(1.0f); };
			constexpr auto sox = init_sox();

			constexpr auto init_six = []() -> auto { if constexpr(IW < OW) return r32(IW) / OW; else return r32(1.0f); };
			constexpr auto six = init_six();

			constexpr auto init_soy = []() -> auto { if constexpr(IH > OH) return r32(OH) / IH; else return r32(1.0f); };
			constexpr auto soy = init_soy();

			constexpr auto init_siy = []() -> auto { if constexpr(IH < OH) return r32(IH) / OH; else return r32(1.0f); };
			constexpr auto siy = init_siy();
			
			auto ox = r32(0.0f);
			auto oy = r32(0.0f);
			
			for(auto iy = r32(0.0f); iy < r32(IH); iy += siy) { for(auto ix = r32(0.0f); ix < r32(IW); ix += six)
			{
				auto Sum = r64(0.0f);
				auto w = u64(0);
				const auto o = math::index(u64(ox), u64(oy), OH);

				for(auto ky = RdxMin; ky != RdxMax; ++ky) { for(auto kx = RdxMin; kx != RdxMax; ++kx)
				{
					const auto px = i64(ix + kx);
					const auto py = i64(iy + ky);
					
					if(this->ShpIn.isInside(px, py))
					{
						Sum += this->Input[this->ShpIn.idx(px, py)] * this->Weights[o][w];
					}

					++w;
				}}

				if constexpr(F == Func::SIGMOID) this->OutTrans[o] = math::sigmoid(Sum);
				if constexpr(F == Func::TANH) this->OutTrans[o] = math::tanh(Sum);
				if constexpr(F == Func::RELU) this->OutTrans[o] = math::relu(Sum);
				if constexpr(F == Func::PRELU) this->OutTrans[o] = math::prelu(Sum, 0.2f);

				if constexpr((F == Func::RELU) || (F == Func::PRELU)) this->OutReal[o] = Sum;

				ox += sox; if(ox >= this->ShpOut[0]) { ox = 0.0f; oy += soy; }
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
			if(!this->IsLocked) for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsDlt[o].clear();
			this->Gradient.clear();

			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target, const r32* _Mask = nullptr ) -> void final
		{
			constexpr auto RdxMin = -i64(R);
			constexpr auto RdxMax = i64(R + 1);

			constexpr auto init_sox = []() -> auto { if constexpr(IW > OW) return r32(OW) / IW; else return r32(1.0f); };
			constexpr auto sox = init_sox();

			constexpr auto init_six = []() -> auto { if constexpr(IW < OW) return r32(IW) / OW; else return r32(1.0f); };
			constexpr auto six = init_six();

			constexpr auto init_soy = []() -> auto { if constexpr(IH > OH) return r32(OH) / IH; else return r32(1.0f); };
			constexpr auto soy = init_soy();

			constexpr auto init_siy = []() -> auto { if constexpr(IH < OH) return r32(IH) / OH; else return r32(1.0f); };
			constexpr auto siy = init_siy();
			
			auto ox = r32(0.0f);
			auto oy = r32(0.0f);
			

			for(auto iy = r32(0.0f); iy < r32(IH); iy += siy) { for(auto ix = r32(0.0f); ix < r32(IW); ix += six)
			{
				const auto o = this->ShpOut.idx(u64(ox), u64(oy));
					
				auto w = u64(0);
					
				auto DerOut = r32(0.0f);
				if(this->Front) DerOut = this->Front->gradient()[o];
				else DerOut = (this->OutTrans[o] - _Target[o]);

				auto DerIn = r32(0.0f);
				if constexpr(F == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[o]) * DerOut;
				if constexpr(F == Func::RELU) DerIn = math::reluDer(this->OutReal[o]) * DerOut;
				if constexpr(F == Func::PRELU) DerIn = math::preluDer(this->OutReal[o], 0.2f) * DerOut;

				for(auto ky = RdxMin; ky != RdxMax; ++ky) { for(auto kx = RdxMin; kx != RdxMax; ++kx)
				{
					const auto px = i64(ix + kx);
					const auto py = i64(iy + ky);

					if(this->ShpIn.isInside(px, py))
					{
						if(!this->IsLocked) this->WeightsDlt[o][w] += (this->Input[this->ShpIn.idx(px, py)] * DerIn);
						this->Gradient[o] += this->Weights[o][w] * DerIn;
					}

					++w;
				}}

				ox += sox; if(ox >= this->ShpOut[0]) { ox = 0.0f; oy += soy; }
			}}


			if(this->Back) this->Back->fit(nullptr, nullptr);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r32 _Rate = 0.01f ) -> void final
		{
			if(!this->IsLocked)
			{
				auto Rate = _Rate;
				if constexpr((F == Func::RELU) || (F == Func::PRELU)) Rate *= 0.1f;
				
				for(auto o = u64(0); o < this->ShpOut.size(); ++o)
				{
					for(auto l = u64(0); l < this->KerArea; ++l)
					{
						this->Weights[o][l] -= this->WeightsDlt[o][l] * Rate;
					}
				}
			}

			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store operation's structure and weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			auto Id = this->id();

			_Stream.write(reinterpret_cast<const char*>(&Id), sizeof(Id));
			_Stream.write(reinterpret_cast<const char*>(&this->KerRadius), sizeof(this->KerRadius));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpIn), sizeof(this->ShpIn));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpOut), sizeof(this->ShpOut));
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) _Stream.write(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());

			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load operation's structure and weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			auto StreamId = u64(0);
			auto StreamKerRadius = u64(0);
			auto StreamShpIn = utl::Shape();
			auto StreamShpOut = utl::Shape();
			
			_Stream.read(reinterpret_cast<char*>(&StreamId), sizeof(StreamId));
			if(this->id() != StreamId) throw Error("sx", "OpLocal2", "load", ERR_LOAD, "Id mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamKerRadius), sizeof(StreamKerRadius));
			if(this->KerRadius != StreamKerRadius) throw Error("sx", "OpLocal2", "load", ERR_LOAD, "Radius mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpIn), sizeof(StreamShpIn));
			if(this->ShpIn != StreamShpIn) throw Error("sx", "OpLocal2", "load", ERR_LOAD, "Input shape mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpOut), sizeof(StreamShpOut));
			if(this->ShpOut != StreamShpOut) throw Error("sx", "OpLocal2", "load", ERR_LOAD, "Output shape mismatch!");
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) _Stream.read(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());

			if(this->Front) this->Front->load(_Stream);
		}
	};
}
