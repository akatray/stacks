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
// Stacks.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Classic dense network.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class OpDense : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Data.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Buffer<r32> OutputFinal;
		Buffer<r32> OutputRaw;
		Buffer<Buffer<r32>> Weights;
		Buffer<Buffer<r32>> WeightsD;
		Buffer<r32> Biases;
		Buffer<r32> BiasesD;
		Buffer<r32> Gradient;
		
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpDense ( const math::Shape& _ShpIn, const math::Shape& _ShpOut, Op* _Last = nullptr ) : Op(_ShpIn, _ShpOut)
		{
			if(_Last)
			{
				this->Last = _Last;
				this->Last->setNext(this);
				this->Input = this->Last->output();
			}

			this->OutputFinal.resize(this->ShpOut.size(), simd::AllocSimd);
			this->OutputRaw.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Output = this->OutputFinal();

			this->Weights.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) {this->Weights[o].resize(this->ShpIn.size(), simd::AllocSimd); rngBuffer(this->Weights[o](), this->Weights[o].size(), -0.001f, 0.001f);}
			this->WeightsD.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsD[o].resize(this->ShpIn.size(), simd::AllocSimd);

			this->Biases.resize(this->ShpOut.size(), simd::AllocSimd); rngBuffer(this->Biases(), this->Biases.size(), -0.1f, 0.1f);
			this->BiasesD.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Gradient.resize(this->ShpIn.size(), simd::AllocSimd);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpDense ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Identify operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto id ( void ) const -> OpId final { return OpId::DENSE; }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute stack for input. Returns pointer to output buffer. Buffer belongs to last operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				this->OutputFinal[o] = math::sigmoid(simd::mulVecByVecSum(this->ShpIn.size(), this->Input, this->Weights[o]()) + this->Biases[o]);
			}

			if(this->Last) this->Input = InputCopy;
			if(this->Next) return this->Next->execute(nullptr);
			else return this->Output;
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

				if(this->Next)
				{
					if(this->Next->id() == OpId::DENSE)
					{
						DerOut = dynamic_cast<OpDense*>(this->Next)->Gradient[o];
					}
				}

				else
				{
					DerOut = this->OutputFinal[o] - _Target[o];
				}

				float DerIn = 0.0f;
				DerIn = math::sigmoidDer2(this->OutputFinal[o]) * DerOut;
				DerIn *= (1.0f + (0.25f * _Depth)); // Boosting for deeper operations. Needs more testing to see if it is of any use.

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
		virtual auto store ( std::ostream& _Stream ) const -> void
		{
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
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				_Stream.read(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());
			}

			_Stream.read(this->Biases.cast<char>(), this->Biases.sizeInBytes());

			if(this->Next) this->Next->load(_Stream);
		}
	};
}
