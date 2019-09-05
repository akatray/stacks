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
	constexpr auto OpFlagLocal2Id = u32(0xC0000000);
	constexpr auto OpFlagLocal2Ver = u32(0x00000200);

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Local dense network 2D.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class OpLocal2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Data.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		u64 Radius;
		u64 LocalSize;
		Buffer<r32> Output;
		Buffer<r32> OutputRaw;
		Buffer<r32> Gradient;
		Buffer<Buffer<r32>> Weights;
		Buffer<Buffer<r32>> WeightsD;
		
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpLocal2 ( const utl::Shape& _ShpIn, const utl::Shape& _ShpOut, const u64 _Radius, Op* _Last = nullptr ) : Op(_ShpIn, _ShpOut)
		{
			this->setLast(_Last);
			
			this->Radius = _Radius;
			this->LocalSize = ((this->Radius * 2) + 1) * ((this->Radius * 2) + 1);

			this->Output.resize(this->ShpOut.size(), simd::AllocSimd);
			this->OutputRaw.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Gradient.resize(this->ShpIn.size(), simd::AllocSimd);

			this->Weights.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) {this->Weights[o].resize(this->LocalSize, simd::AllocSimd); rngBuffer(this->Weights[o](), this->Weights[o].size(), 0.001f, 0.01f);}
			this->WeightsD.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsD[o].resize(this->LocalSize, simd::AllocSimd);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpLocal2 ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial Set/Get functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto flags ( void ) const -> u32 final { return (OpFlagLocal2Id | OpFlagLocal2Ver | OpFlagTraitReal); }
		auto output ( void ) -> r32* final { return this->Output(); }
		auto gradient ( void ) const -> const r32* final { return this->Gradient(); }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute stack for input. Returns pointer to output buffer. Buffer belongs to last operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			
			auto ItrIn = utl::ShapeIter<r32>(this->ShpIn, r32(this->ShpIn[0]) / this->ShpOut[0], r32(this->ShpIn[1]) / this->ShpOut[1], 1.0f);
			auto ItrOut = utl::ShapeIter<u64>(this->ShpOut, 1, 1, 1);
			const auto RadiusMin = -i64(this->Radius);
			const auto RadiusMax = i64(this->Radius + 1);

			while(!ItrIn.isDone())
			{
				auto Sum = r32(0.0f);
				auto w = u64(0);

				for(auto yf = RadiusMin; yf != RadiusMax; ++yf) { for(auto xf = RadiusMin; xf != RadiusMax; ++xf)
				{
					if(this->ShpIn.isInside(ItrIn[0] + xf, ItrIn[1] + yf))
					{
						Sum += this->Input[this->ShpIn.idx(ItrIn[0] + xf, ItrIn[1] + yf, ItrIn[2])] * this->Weights[ItrOut.idx()][w];
					}

					++w;
				}}

				this->OutputRaw[ItrOut.idx()] = Sum;
				this->Output[ItrOut.idx()] = math::prelu(Sum);
				
				ItrIn.next();
				ItrOut.next();
			}

			if(this->Last) this->Input = InputCopy;
			if(this->Next) return this->Next->execute(nullptr);
			else return this->Output();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Setup stack for fit().
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		virtual auto reset ( void ) -> void
		{
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsD[o].clear();
			this->Gradient.clear();

			if(this->Next) this->Next->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Back propagate target through stack. Needs to have input executed first.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target, const u64 _Depth = 0 ) -> void final
		{
			auto ItrIn = utl::ShapeIter<r32>(this->ShpIn, r32(this->ShpIn[0]) / this->ShpOut[0], r32(this->ShpIn[1]) / this->ShpOut[1], 1.0f);
			auto ItrOut = utl::ShapeIter<u64>(this->ShpOut, 1, 1, 1);
			const auto RadiusMin = -i64(this->Radius);
			const auto RadiusMax = i64(this->Radius + 1);

			while(!ItrIn.isDone())
			{
				float DerOut = 0.0f;
				if(this->Next) DerOut = this->Next->gradient()[ItrOut.idx()];
				else DerOut = this->Output[ItrOut.idx()] - _Target[ItrOut.idx()];

				float DerIn = 0.0f;
				DerIn = math::preluDer(this->OutputRaw[ItrOut.idx()]) * DerOut;

				auto w = u64(0);

				for(auto yf = RadiusMin; yf != RadiusMax; ++yf) { for(auto xf = RadiusMin; xf != RadiusMax; ++xf)
				{
					if(this->ShpIn.isInside(ItrIn[0] + xf, ItrIn[1] + yf))
					{
						this->WeightsD[ItrOut.idx()][w] += Input[this->ShpIn.idx(ItrIn[0] + xf, ItrIn[1] + yf, ItrIn[2])] * DerIn;
						this->Gradient[ItrIn.idx()] += this->Weights[ItrOut.idx()][w] * DerIn;
					}

					++w;
				}}

				ItrIn.next();
				ItrOut.next();
			}

			if(this->Last) this->Last->fit(nullptr, _Depth + 1);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply deltas generated by fit() to stack.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r32 _Rate = 0.01f ) -> void final
		{
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				for(auto l = u64(0); l < this->LocalSize; ++l)
				{
					this->Weights[o][l] -= this->WeightsD[o][l] * (_Rate * 0.01f); // prelu explodes on high rates.
				}

			}

			if(this->Next) this->Next->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store stack's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			auto Flags = this->flags();

			_Stream.write(reinterpret_cast<const char*>(&Flags), sizeof(Flags));
			_Stream.write(reinterpret_cast<const char*>(&this->Radius), sizeof(this->Radius));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpIn), sizeof(this->ShpIn));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpOut), sizeof(this->ShpOut));
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) _Stream.write(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());

			if(this->Next) this->Next->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load stack's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
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
		}
	};
}
