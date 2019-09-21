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
	class OpLocal2 : public Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Func Trans;
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
		OpLocal2 ( const utl::Shape& _ShpIn, const utl::Shape& _ShpOut, const u64 _Radius = 1, const Func _Trans = Func::TANH, r32 _IntMin = 0.001f, r32 _IntMax = 0.002f ) : Op(_ShpIn, _ShpOut)
		{
			if((_ShpIn[2] != 1) || (_ShpOut[2] != 1))
			{
				auto Description = std::stringstream();
				Description << "Input/Output shape must have depth of 1!";
				Description << "Current: In(" << _ShpIn[2] << "), Out(" << _ShpOut[2] << ").";
				throw Error("sx", "OpLocal2", "Constructor", ERR_BAD_SHAPE, Description.str());
			}
			
			this->Trans = _Trans;
			this->KerRadius = _Radius;
			this->KerArea = ((_Radius * 2) + 1) * ((_Radius * 2) + 1);

			this->OutTrans.resize(this->ShpOut.size(), simd::AllocSimd);
			if((_Trans == Func::RELU) || (_Trans == Func::PRELU)) this->OutReal.resize(this->ShpOut.size(), simd::AllocSimd);
			this->Gradient.resize(this->ShpIn.size(), simd::AllocSimd);

			this->Weights.resize(this->ShpOut.size(), simd::AllocSimd);
			for(auto o = u64(0); o < this->ShpOut.size(); ++o) {this->Weights[o].resize(this->KerArea, simd::AllocSimd); rngBuffer(this->Weights[o](), this->Weights[o].size(), _IntMin, _IntMax);}
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
			
			auto ItrIn = utl::ShapeIter<r32>(this->ShpIn, r32(this->ShpIn[0]) / this->ShpOut[0], r32(this->ShpIn[1]) / this->ShpOut[1], 1.0f);
			auto ItrOut = utl::ShapeIter<u64>(this->ShpOut);
			const auto RadiusMin = -i64(this->KerRadius);
			const auto RadiusMax = i64(this->KerRadius + 1);

			while(!ItrIn.isDone())
			{
				const auto IdxOut = ItrOut.idx();
				auto Sum = r32(0.0f);
				auto IdxW = u64(0);

				for(auto yf = RadiusMin; yf != RadiusMax; ++yf) { for(auto xf = RadiusMin; xf != RadiusMax; ++xf)
				{
					if(this->ShpIn.isInside(i64(ItrIn[0] + xf), i64(ItrIn[1] + yf)))
					{
						Sum += this->Input[this->ShpIn.idx(u64(ItrIn[0] + xf), u64(ItrIn[1] + yf), u64(ItrIn[2]))] * this->Weights[IdxOut][IdxW];
					}

					++IdxW;
				}}
				
				if((this->Trans == Func::RELU) || (this->Trans == Func::PRELU)) this->OutReal[IdxOut] = Sum;

				if(this->Trans == Func::SIGMOID) this->OutTrans[IdxOut] = math::sigmoid(Sum);
				if(this->Trans == Func::TANH) this->OutTrans[IdxOut] = math::tanh(Sum);
				if(this->Trans == Func::RELU) this->OutTrans[IdxOut] = math::relu(Sum);
				if(this->Trans == Func::PRELU) this->OutTrans[IdxOut] = math::prelu(Sum);
				
				ItrIn.next();
				ItrOut.next();
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
			if(!this->IsLocked) for(auto o = u64(0); o < this->ShpOut.size(); ++o) this->WeightsDlt[o].clear();
			this->Gradient.clear();

			if(this->Front) this->Front->reset();
		}
		
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Fit target.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto fit ( const r32* _Target, const u64 _Depth = 0 ) -> void final
		{
			auto ItrIn = utl::ShapeIter<r32>(this->ShpIn, r32(this->ShpIn[0]) / this->ShpOut[0], r32(this->ShpIn[1]) / this->ShpOut[1], 1.0f);
			auto ItrOut = utl::ShapeIter<u64>(this->ShpOut, 1, 1, 1);
			const auto RadiusMin = -i64(this->KerRadius);
			const auto RadiusMax = i64(this->KerRadius + 1);

			while(!ItrIn.isDone())
			{
				const auto IdxOut = ItrOut.idx();
				
				float DerOut = 0.0f;
				if(this->Front) DerOut = this->Front->gradient()[IdxOut];
				else DerOut = this->OutTrans[IdxOut] - _Target[IdxOut];

				float DerIn = 0.0f;
				if(this->Trans == Func::SIGMOID) DerIn = math::sigmoidDer2(this->OutTrans[IdxOut]) * DerOut;
				if(this->Trans == Func::TANH) DerIn = math::tanhDer2(this->OutTrans[IdxOut]) * DerOut;
				if(this->Trans == Func::RELU) DerIn = math::reluDer(this->OutReal[IdxOut]) * DerOut;
				if(this->Trans == Func::PRELU) DerIn = math::preluDer(this->OutReal[IdxOut]) * DerOut;

				auto IdxW = u64(0);

				for(auto yf = RadiusMin; yf != RadiusMax; ++yf) { for(auto xf = RadiusMin; xf != RadiusMax; ++xf)
				{
					if(this->ShpIn.isInside(ItrIn[0] + xf, ItrIn[1] + yf))
					{
						if(!this->IsLocked) this->WeightsDlt[IdxOut][IdxW] += Input[this->ShpIn.idx(ItrIn[0] + xf, ItrIn[1] + yf, ItrIn[2])] * DerIn;
						this->Gradient[ItrIn.idx()] += this->Weights[IdxOut][IdxW] * DerIn;
					}

					++IdxW;
				}}

				ItrIn.next();
				ItrOut.next();
			}

			if(this->Back) this->Back->fit(nullptr, _Depth + 1);
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Apply deltas.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto apply ( const r32 _Rate = 0.01f ) -> void final
		{
			if(!this->IsLocked)
			{
				auto Rate = _Rate;
				if((this->Trans == Func::RELU) || (this->Trans == Func::PRELU)) Rate *= 0.1f;
				
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
