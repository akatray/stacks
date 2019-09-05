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
	constexpr auto OpFlagDenseId = u32(0xB0000000);
	constexpr auto OpFlagDenseVer = u32(0x00000200);

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Classic dense network.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class OpDense : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Data.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Buffer<r32> Output;
		Buffer<r32> Gradient;
		Buffer<Buffer<r32>> Weights;
		Buffer<Buffer<r32>> WeightsD;
		Buffer<r32> Biases;
		Buffer<r32> BiasesD;
		
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpDense ( const utl::Shape& _ShpIn, const utl::Shape& _ShpOut, Op* _Last = nullptr ) : Op(_ShpIn, _ShpOut)
		{
			this->setLast(_Last);

			this->Output.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Gradient.resize(this->ShpIn.size(), simd::AllocSimd);

			this->Weights.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) {this->Weights[o].resize(this->ShpIn.size(), simd::AllocSimd); rngBuffer(this->Weights[o](), this->Weights[o].size(), -0.001f, 0.001f);}
			this->WeightsD.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsD[o].resize(this->ShpIn.size(), simd::AllocSimd);

			this->Biases.resize(this->ShpOut.size(), simd::AllocSimd); rngBuffer(this->Biases(), this->Biases.size(), -0.1f, 0.1f);
			this->BiasesD.resize(this->ShpOut.size(), simd::AllocSimd);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpDense ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial Set/Get functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto flags ( void ) const -> u32 final { return (OpFlagDenseId | OpFlagDenseVer | OpFlagTraitReal); }
		auto output ( void ) -> r32* final { return this->Output(); }
		auto gradient ( void ) const -> const r32* final { return this->Gradient(); }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute stack for input. Returns pointer to output buffer. Buffer belongs to last operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				this->Output[o] = math::sigmoid(simd::mulVecByVecSum(this->ShpIn.size(), this->Input, this->Weights[o]()) + this->Biases[o]);
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
			this->BiasesD.clear();
			this->Gradient.clear();

			if(this->Next) this->Next->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Back propagate target through stack. Needs to have input executed first.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target, const u64 _Depth = 0 ) -> void final
		{
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				float DerOut = 0.0f;

				if(this->Next) DerOut = this->Next->gradient()[o];
				else DerOut = this->Output[o] - _Target[o];

				float DerIn = 0.0f;
				DerIn = math::sigmoidDer2(this->Output[o]) * DerOut;
				//DerIn *= (1.0f + (0.25f * _Depth)); // Boosting for deeper operations. Needs more testing to see if it is of any use.

				simd::mulVecByConstAddToOut(this->ShpIn.size(), this->Gradient(), this->Weights[o](), DerIn);
				simd::mulVecByConstAddToOut(this->ShpIn.size(), this->WeightsD[o](), this->Input, DerIn);
				this->BiasesD[o] += DerIn;
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
				simd::mulVecByConstSubFromOut(this->ShpIn.size(), this->Weights[o](), this->WeightsD[o](), _Rate);
			}

			simd::mulVecByConstSubFromOut(this->ShpOut.size(), this->Biases(), this->BiasesD(), _Rate);

			if(this->Next) this->Next->apply(_Rate);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store stack's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto store ( std::ostream& _Stream ) const -> void final
		{
			auto Flags = this->flags();

			_Stream.write(reinterpret_cast<const char*>(&Flags), sizeof(Flags));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpIn), sizeof(this->ShpIn));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpOut), sizeof(this->ShpOut));
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				_Stream.write(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());
			}

			_Stream.write(this->Biases.cast<char>(), this->Biases.sizeInBytes());

			if(this->Next) this->Next->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load stack's weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		virtual auto load ( std::istream& _Stream ) -> void
		{
			auto StreamFlags = u32(0);
			auto StreamShpIn = utl::Shape();
			auto StreamShpOut = utl::Shape();
			
			_Stream.read(reinterpret_cast<char*>(&StreamFlags), sizeof(StreamFlags));
			if(this->flags() != StreamFlags) throw str("sx->OpDense->load(): Flags mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpIn), sizeof(StreamShpIn));
			if(this->ShpIn != StreamShpIn) throw str("sx->OpDense->load(): Input shape mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpOut), sizeof(StreamShpOut));
			if(this->ShpOut != StreamShpOut) throw str("sx->OpDense->load(): Output shape mismatch!");
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				_Stream.read(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());
			}

			_Stream.read(this->Biases.cast<char>(), this->Biases.sizeInBytes());

			if(this->Next) this->Next->load(_Stream);
		}
	};
}
