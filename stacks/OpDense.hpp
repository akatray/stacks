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
	// Classic dense network.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class OpDense : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Data.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Func Trans;
		Buffer<r32> OutTrans;
		Buffer<r32> OutReal;
		Buffer<r32> Gradient;
		Buffer<Buffer<r32>> Weights;
		Buffer<Buffer<r32>> WeightsDlt;
		Buffer<r32> Biases;
		Buffer<r32> BiasesDlt;
		
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Explicit constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		OpDense ( const utl::Shape& _ShpIn, const utl::Shape& _ShpOut, const Func _Trans = Func::SIGMOID, r32 _IntMin = -0.001f, r32 _IntMax = 0.001f ) : Op(_ShpIn, _ShpOut)
		{
			this->OutTrans.resize(this->ShpOut.size(), simd::AllocSimd);
			if((_Trans == Func::RELU) || (_Trans == Func::PRELU)) this->OutReal.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Gradient.resize(this->ShpIn.size(), simd::AllocSimd);

			this->Weights.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) {this->Weights[o].resize(this->ShpIn.size(), simd::AllocSimd); rng::rbuf(this->Weights[o](), this->Weights[o].size(), _IntMin, _IntMax);}
			this->WeightsDlt.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsDlt[o].resize(this->ShpIn.size(), simd::AllocSimd);

			this->Biases.resize(this->ShpOut.size(), simd::AllocSimd); rng::rbuf(this->Biases(), this->Biases.size(), _IntMin, _IntMax);
			this->BiasesDlt.resize(this->ShpOut.size(), simd::AllocSimd);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Virtual destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~OpDense ( void ) final {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Trivial functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto id ( void ) const -> u64 final { return 1000; }
		auto output ( void ) -> r32* final { return this->OutTrans(); }
		auto gradient ( void ) const -> const r32* final { return this->Gradient(); }

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Execute operation.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto execute ( const r32* _Input ) -> r32* final
		{
			auto InputCopy = this->Input;
			if(_Input) this->Input = _Input;
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				auto Sum = simd::mulVecByVecSum(this->ShpIn.size(), this->Input, this->Weights[o]()) + this->Biases[o];
				
				if((this->Trans == Func::RELU) || (this->Trans == Func::PRELU)) this->OutReal[o] = Sum;
				
				if(this->Trans == Func::SIGMOID) this->OutTrans[o] = math::sigmoid(Sum);
				else if(this->Trans == Func::TANH) this->OutTrans[o] = math::tanh(Sum);
				else if(this->Trans == Func::RELU) this->OutTrans[o] = math::relu(Sum);
				else if(this->Trans == Func::PRELU) this->OutTrans[o] = math::crelu(Sum);
			}

			if(this->Back) this->Input = InputCopy;
			if(this->Front) return this->Front->execute(nullptr);
			else return this->output();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Reset deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		virtual auto reset ( void ) -> void
		{
			if(!this->IsLocked)
			{
				for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsDlt[o].clear();
				this->BiasesDlt.clear();
			}

			this->Gradient.clear();

			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target, const r32* _Mask = nullptr ) -> void final
		{
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				auto DerOut = r32(0.0f);

				if(this->Front) DerOut = this->Front->gradient()[o];
				else DerOut = this->OutTrans[o] - _Target[o];

				auto DerIn = r32(0.0f);
				if(this->Trans == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[o]) * DerOut;
				if(this->Trans == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[o]) * DerOut;
				if(this->Trans == Func::RELU) DerIn = math::reluDer(this->OutReal[o]) * DerOut;
				if(this->Trans == Func::PRELU) DerIn = math::creluDer(this->OutReal[o]) * DerOut;

				simd::mulVecByConstAddToOut(this->ShpIn.size(), this->Gradient(), this->Weights[o](), DerIn);
				if(!this->IsLocked) simd::mulVecByConstAddToOut(this->ShpIn.size(), this->WeightsDlt[o](), this->Input, DerIn);
				if(!this->IsLocked) this->BiasesDlt[o] += DerIn;
			}

			if(this->Back) this->Back->fit(nullptr, nullptr);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r32 _Rate = 0.01f ) -> void final
		{
			if(!this->IsLocked)
			{
				for(auto o = u64(0); o < this->ShpOut.size(); ++o)
				{
					simd::mulVecByConstSubFromOut(this->ShpIn.size(), this->Weights[o](), this->WeightsDlt[o](), _Rate);
				}

				simd::mulVecByConstSubFromOut(this->ShpOut.size(), this->Biases(), this->BiasesDlt(), _Rate);
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
			_Stream.write(reinterpret_cast<const char*>(&this->ShpIn), sizeof(this->ShpIn));
			_Stream.write(reinterpret_cast<const char*>(&this->ShpOut), sizeof(this->ShpOut));
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				_Stream.write(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());
			}

			_Stream.write(this->Biases.cast<char>(), this->Biases.sizeInBytes());

			if(this->Front) this->Front->store(_Stream);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load operation's structure and weights.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto load ( std::istream& _Stream ) -> void final
		{
			auto StreamId = u64(0);
			auto StreamShpIn = utl::Shape();
			auto StreamShpOut = utl::Shape();
			
			_Stream.read(reinterpret_cast<char*>(&StreamId), sizeof(StreamId));
			if(this->id() != StreamId) throw Error("sx", "OpDense", "load", ERR_LOAD, "Id mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpIn), sizeof(StreamShpIn));
			if(this->ShpIn != StreamShpIn) throw Error("sx", "OpDense", "load", ERR_LOAD, "Input shape mismatch!");

			_Stream.read(reinterpret_cast<char*>(&StreamShpOut), sizeof(StreamShpOut));
			if(this->ShpOut != StreamShpOut) throw Error("sx", "OpDense", "load", ERR_LOAD, "Output shape mismatch!");
			
			for(auto o = u64(0); o < this->ShpOut.size(); ++o)
			{
				_Stream.read(this->Weights[o].cast<char>(), this->Weights[o].sizeInBytes());
			}

			_Stream.read(this->Biases.cast<char>(), this->Biases.sizeInBytes());

			if(this->Front) this->Front->load(_Stream);
		}
	};
}
