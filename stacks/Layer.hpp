// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include "./Error.hpp"
#include "./Transfer.hpp"
#include "./Optimizer.hpp"

#include "./layer/data/Outputs.hpp"
#include "./layer/data/Weights.hpp"
#include "./layer/data/Biases.hpp"

#include <iostream>

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Neural Networks Experiment.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Nothing.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	struct Nothing {};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Macros.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function signatures.
	#define SX_FNSIG_LAYER_OUTSZ auto outSz ( void ) const -> uMAX
	#define SX_FNSIG_LAYER_OUTSZBT auto outSzBt ( void ) const -> uMAX
	#define SX_FNSIG_LAYER_OUT auto out ( void ) const -> const T*
	#define SX_FNSIG_LAYER_GRAD auto gradient ( void ) const -> const T*
	#define SX_FNSIG_LAYER_EXE auto exe ( const bool _Chain = true ) -> void
	#define SX_FNSIG_LAYER_RESET auto reset ( const bool _Chain = true ) -> void
	#define SX_FNSIG_LAYER_ERR auto err ( const T* _Target ) -> T
	#define SX_FNSIG_LAYER_FIT auto fit ( const T* _Target, const rMAX _ErrParam, const bool _Chain = true ) -> void
	#define SX_FNSIG_LAYER_APPLY auto apply ( const rMAX _Rate, const uMAX _Iter = 0, const bool _Chain = true ) -> void
	#define SX_FNSIG_LAYER_STORE auto store ( std::ostream& _Stream, const bool _Chain = true ) const -> void
	#define SX_FNSIG_LAYER_LOAD auto load ( std::istream& _Stream, const bool _Chain = true ) -> void
	#define SX_FNSIG_LAYER_EXCHANGE auto exchange ( Layer<T>* _Master, const bool _Chain = true ) -> void
	
	// Macros for chained function calls.
	#define SX_MC_LAYER_NEXT_EXE if(this->Front && _Chain) this->Front->exe()
	#define SX_MC_LAYER_NEXT_RESET if(this->Front && _Chain) this->Front->reset()
	#define SX_MC_LAYER_NEXT_FIT if(this->Back && _Chain) this->Back->fit(nullptr, _ErrParam)
	#define SX_MC_LAYER_NEXT_APPLY if(this->Front && _Chain) this->Front->apply(_Rate, _Iter)
	#define SX_MC_LAYER_NEXT_STORE if(this->Front && _Chain) this->Front->store(_Stream)
	#define SX_MC_LAYER_NEXT_LOAD if(this->Front && _Chain) this->Front->load(_Stream)

	// Generate code for trivial functions.
	#define SX_MC_LAYER_TRIVIAL(CLASS_NAME, SZ_OUT, PTR_OUT, PTR_GRAD) public: ~CLASS_NAME ( void ) final {} constexpr SX_FNSIG_LAYER_OUTSZ final { return SZ_OUT; } constexpr SX_FNSIG_LAYER_OUTSZBT final { return SZ_OUT * sizeof(T); } constexpr SX_FNSIG_LAYER_OUT final { return PTR_OUT; } constexpr SX_FNSIG_LAYER_GRAD final { return PTR_GRAD; }

	// Generate code common derivatives.
	#define SX_MC_LAYER_DER_ERR auto DerErr = T(0); if(this->Front) DerErr = this->Front->gradient()[o]; else DerErr += errorDer<T,FN_ERR>(_Target[o], this->OutTrans[o])
	#define SX_MC_LAYER_DER_TRANS auto ValRaw = T(); if constexpr(needRaw<T,FN_TRANS>()) ValRaw = this->OutRaw[o]; auto DerTrans = transferDer<T,FN_TRANS>(this->OutTrans[o], ValRaw) * DerErr; DerTrans = std::clamp(DerTrans, T(-1), T(1))
	
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Layer interface.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> class Layer
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		protected:
		Layer* Back;
		Layer* Front;
		const T* Input;
		bool IsLocked;
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Layer ( void ) : Front(nullptr), Back(nullptr), Input(nullptr), IsLocked(false) {}
		virtual ~Layer ( void ) {}

		virtual SX_FNSIG_LAYER_OUTSZ = 0; // Get output size in Ts.
		virtual SX_FNSIG_LAYER_OUTSZBT = 0; // Get output size in chars.
		virtual SX_FNSIG_LAYER_OUT = 0; // Get output buffer pointer.
		virtual SX_FNSIG_LAYER_GRAD = 0; // Get gradient buffer pointer.

		virtual SX_FNSIG_LAYER_EXE = 0; // Execute.
		virtual SX_FNSIG_LAYER_FIT = 0; // Backpropagate.

		virtual SX_FNSIG_LAYER_ERR { return 0; } // Get output error in respect to argument.
		virtual SX_FNSIG_LAYER_RESET { if(this->Front) this->Front->reset(); } // Reset delta parameters.
		virtual SX_FNSIG_LAYER_APPLY { if(this->Front) this->Front->apply(_Rate, _Iter); } // Apply optimizations and update parameters.
		virtual SX_FNSIG_LAYER_STORE { if(this->Front) this->Front->store(_Stream); } // Store parameters to stream.
		virtual SX_FNSIG_LAYER_LOAD { if(this->Front) this->Front->load(_Stream); } // Load parameters from stream.
		virtual SX_FNSIG_LAYER_EXCHANGE = 0; // Multi threading utility.

		inline auto lock ( void ) -> void { this->IsLocked = true; }
		inline auto unlock ( void ) -> void { this->IsLocked = false; }
		inline auto back ( void ) -> Layer* { return this->Back; }
		inline auto front ( void ) -> Layer* { return this->Front; }
		inline auto back ( void ) const -> const Layer* { return this->Back; }
		inline auto front ( void ) const -> const Layer* { return this->Front; }
		inline auto setBack ( Layer* _Back ) -> void { this->Back = _Back; if(_Back) { this->Input = _Back->out(); _Back->setFront(this); } }
		inline auto setFront ( Layer* _Front ) -> void { this->Front = _Front; }
		inline auto setInput ( const T* _Input ) -> const T* { const auto InputLast = this->Input; if(_Input) this->Input = _Input; return InputLast; }
	};
}
