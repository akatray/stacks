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
	// Operation flags.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//constexpr auto OpFlagLocal2Id = u32(0xC0000000);
	//constexpr auto OpFlagLocal2Ver = u32(0x00000200);

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Local dense network 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class OpConv2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Data.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Func Trans;
		u64 KerRadius;
		u64 KerArea;
		u64 KerCount;
		Buffer<r32> OutTrans;
		Buffer<r32> OutReal;
		Buffer<r32> Gradient;
		Buffer<Buffer<Buffer<r64>>> Weights;
		Buffer<Buffer<Buffer<r64>>> WeightsDlt;
		
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpConv2 ( const utl::Shape& _ShpIn, const utl::Shape& _ShpOut, const u64 _Radius, const Func _Trans = Func::PRELU, r32 _IntMin = 0.001f, r32 _IntMax = 0.002f ) : Op(_ShpIn, _ShpOut)
		{
			this->Trans = _Trans;
			this->KerRadius = _Radius;
			this->KerArea = (((_Radius * 2) + 1) * ((_Radius * 2) + 1));
			this->KerCount = this->ShpOut[2];

			this->OutTrans.resize(this->ShpOut.size(), simd::AllocSimd);
			if((_Trans == Func::RELU) || (_Trans == Func::PRELU)) this->OutReal.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Gradient.resize(this->ShpIn.size(), simd::AllocSimd);

			this->Weights.resize(this->KerCount, simd::AllocSimd);
			this->WeightsDlt.resize(this->KerCount, simd::AllocSimd);
			
			for(auto k = u64(0); k < this->KerCount; ++k)
			{
				this->Weights[k].resize(this->ShpIn[2], simd::AllocSimd);
				this->WeightsDlt[k].resize(this->ShpIn[2], simd::AllocSimd);

				for(auto d = u64(0); d < this->ShpIn[2]; ++d)
				{
					this->Weights[k][d].resize(this->KerArea, simd::AllocSimd);
					this->WeightsDlt[k][d].resize(this->KerArea, simd::AllocSimd);

					rng::rbuf<r64>(this->Weights[k][d](), this->Weights[k][d].size(), _IntMin, _IntMax);
				}
			}
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpConv2 ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial Set/Get functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto id ( void ) const -> u64 final { return 3000; }
		auto output ( void ) -> r32* final { return this->OutTrans(); }
		auto gradient ( void ) const -> const r32* final { return this->Gradient(); }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute stack for input. Returns pointer to output buffer. Buffer belongs to last operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;


			const auto RdxMin = -i64(this->KerRadius);
			const auto RdxMax = i64(this->KerRadius + 1);

			auto sox = r32(1.0f);
			if(this->ShpIn[0] > this->ShpOut[0]) sox = r32(this->ShpOut[0]) / this->ShpIn[0];
			
			auto six = r32(1.0f);
			if(this->ShpIn[0] < this->ShpOut[0]) six = r32(this->ShpIn[0]) / this->ShpOut[0];

			auto soy = r32(1.0f);
			if(this->ShpIn[1] > this->ShpOut[1]) soy = r32(this->ShpOut[1]) / this->ShpIn[1];
		
			auto siy = r32(1.0f);
			if(this->ShpIn[1] < this->ShpOut[1]) siy = r32(this->ShpIn[1]) / this->ShpOut[1];
			
			
			for(auto k = u64(0); k < this->KerCount; ++k)
			{
				auto ox = r32(0.0f);
				auto oy = r32(0.0f);
				
				for(auto iy = r32(0.0f); iy < this->ShpIn[1]; iy += siy) { for(auto ix = r32(0.0f); ix < this->ShpIn[0]; ix += six)
				{
					auto Sum = r64(0.0f);
					auto w = u64(0);

					for(auto ky = RdxMin; ky != RdxMax; ++ky) { for(auto kx = RdxMin; kx != RdxMax; ++kx)
					{
						if(this->ShpIn.isInside(i64(ix + kx), i64(iy + ky)))
						{
							for(auto d = u64(0); d < this->ShpIn[2]; ++d)
							{
								Sum += this->Input[this->ShpIn.idx(u64(ix + kx), u64(iy + ky), d)] * this->Weights[k][d][w];
							}
						}

						++w;
					}}

					if(this->Trans == Func::SIGMOID) this->OutTrans[this->ShpOut.idx(u64(ox), u64(oy), k)] = math::sigmoid(Sum);
					if(this->Trans == Func::TANH) this->OutTrans[this->ShpOut.idx(u64(ox), u64(oy), k)] = math::tanh(Sum);
					if(this->Trans == Func::RELU) this->OutTrans[this->ShpOut.idx(u64(ox), u64(oy), k)] = math::relu(Sum);
					if(this->Trans == Func::PRELU) this->OutTrans[this->ShpOut.idx(u64(ox), u64(oy), k)] = math::crelu(Sum, 0.2f);

					if((this->Trans == Func::RELU) || (this->Trans == Func::PRELU)) this->OutReal[this->ShpOut.idx(u64(ox), u64(oy), k)] = Sum;

					ox += sox; if(ox >= this->ShpOut[0]) { ox = 0.0f; oy += soy; }
				}}
			}


			if(this->Back) this->Input = InputCopy;
			if(this->Front) return this->Front->execute(nullptr);
			else return this->output();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Setup stack for fit().
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		virtual auto reset ( void ) -> void
		{
			if(!this->IsLocked)
			{
				for(auto f = u64(0); f < this->KerCount; ++f)
				{
					for(auto d = u64(0); d < this->ShpIn[2]; ++d)
					{
						this->WeightsDlt[f][d].clear();
					}
				}
			}

			this->Gradient.clear();

			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Back propagate target through stack. Needs to have input executed first.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target, const r32* _Mask = nullptr ) -> void final
		{
			const auto RdxMin = -i64(this->KerRadius);
			const auto RdxMax = i64(this->KerRadius + 1);

			auto sox = r32(1.0f);
			if(this->ShpIn[0] > this->ShpOut[0]) sox = r32(this->ShpOut[0]) / this->ShpIn[0];
			
			auto six = r32(1.0f);
			if(this->ShpIn[0] < this->ShpOut[0]) six = r32(this->ShpIn[0]) / this->ShpOut[0];

			auto soy = r32(1.0f);
			if(this->ShpIn[1] > this->ShpOut[1]) soy = r32(this->ShpOut[1]) / this->ShpIn[1];
		
			auto siy = r32(1.0f);
			if(this->ShpIn[1] < this->ShpOut[1]) siy = r32(this->ShpIn[1]) / this->ShpOut[1];
			
			
			for(auto k = u64(0); k < this->KerCount; ++k)
			{
				auto ox = r32(0.0f);
				auto oy = r32(0.0f);
				
				for(auto iy = r32(0.0f); iy < this->ShpIn[1]; iy += siy) { for(auto ix = r32(0.0f); ix < this->ShpIn[0]; ix += six)
				{
					const auto o = this->ShpOut.idx(u64(ox), u64(oy), k);
					
					auto w = u64(0);
					
					auto DerOut = r32(0.0f);
					if(this->Front) DerOut = this->Front->gradient()[o];
					else DerOut = (this->OutTrans[o] - _Target[o]);

					auto DerIn = r32(0.0f);
					if(this->Trans == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[o]) * DerOut;
					if(this->Trans == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[o]) * DerOut;
					if(this->Trans == Func::RELU) DerIn = math::reluDer(this->OutReal[o]) * DerOut;
					if(this->Trans == Func::PRELU) DerIn = math::creluDer(this->OutReal[o], 0.2f) * DerOut;

					DerIn *= (sox * soy) / (this->KerArea * this->KerCount);

					for(auto ky = RdxMin; ky != RdxMax; ++ky) { for(auto kx = RdxMin; kx != RdxMax; ++kx)
					{
						if(this->ShpIn.isInside(i64(ix + kx), i64(iy + ky)))
						{
							for(auto d = u64(0); d < this->ShpIn[2]; ++d)
							{
								if(!this->IsLocked) this->WeightsDlt[k][d][w] += (this->Input[this->ShpIn.idx(u64(ix + kx), u64(iy + ky), d)] * DerIn);
								this->Gradient[this->ShpIn.idx(u64(ix + kx), u64(iy + ky), d)] += this->Weights[k][d][w] * DerIn;
							}
						}

						++w;
					}}

					ox += sox; if(ox >= this->ShpOut[0]) { ox = 0.0f; oy += soy; }
				}}
			}


			if(this->Back) this->Back->fit(nullptr, nullptr);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply deltas generated by fit() to stack.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r32 _Rate = 0.01f ) -> void final
		{
			if(!this->IsLocked)
			{
				auto LocalRate = _Rate;// * 0.1f;
				
				for(auto f = u64(0); f < this->KerCount; ++f)
				{
					for(auto d = u64(0); d < this->ShpIn[2]; ++d)
					{
						for(auto w = u64(0); w < this->KerArea; ++w)
						{
							this->Weights[f][d][w] -= this->WeightsDlt[f][d][w] * LocalRate;
						}
					}
				}
			}

			if(this->Front) this->Front->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store stack's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			/*
			auto Flags = this->flags();

			_Stream.write(reinterpret_cast<const char*>(&Flags), sizeof(Flags));
			_Stream.write(reinterpret_cast<const char*>(&this->Radius), sizeof(this->Radius));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpIn), sizeof(this->ShpIn));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpOut), sizeof(this->ShpOut));
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) _Stream.write(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());

			if(this->Next) this->Next->store(_Stream);
			*/
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load stack's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			/*
			auto StreamFlags = u32(0);
			auto StreamRadius = u64(0);
			auto StreamShpIn = utl::Shape();
			auto StreamShpOut = utl::Shape();
			
			_Stream.read(reinterpret_cast<char*>(&StreamFlags), sizeof(StreamFlags));
			if(this->flags() != StreamFlags) throw str("sx->OpLocal2->load(): Flags mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamRadius), sizeof(StreamRadius));
			if(this->Radius != StreamRadius) throw str("sx->OpLocal2->load(): Radius mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpIn), sizeof(StreamShpIn));
			if(this->ShpIn != StreamShpIn) throw str("sx->OpLocal2->load(): Input shape mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpOut), sizeof(StreamShpOut));
			if(this->ShpOut != StreamShpOut) throw str("sx->OpLocal2->load(): Output shape mismatch!");
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) _Stream.read(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());

			if(this->Next) this->Next->load(_Stream);
			*/
		}
	};
}
