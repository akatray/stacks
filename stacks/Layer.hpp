// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <fx/Types.hpp>
#include "./fn_error.hpp"
#include "./fn_transfer.hpp"
#include "./Optimizer.hpp"
#include "./fn_vector.hpp"
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
	// Macros.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#define SX_FNSIG_LAYER_EXE auto exe ( void ) -> void
	#define SX_FNSIG_LAYER_RESET auto reset ( void ) -> void
	#define SX_FNSIG_LAYER_ERR auto err (const T* _Target) -> T
	#define SX_FNSIG_LAYER_FIT auto fit ( const T* _Target, const T _Rate, const T _ErrParam ) -> void
	#define SX_FNSIG_LAYER_APPLY auto apply ( const r64 _Rate ) -> void
	#define SX_FNSIG_LAYER_STORE auto store ( std::ostream& _Stream ) const -> void
	#define SX_FNSIG_LAYER_LOAD auto load ( std::istream& _Stream ) -> void
	
	
	#define SX_MC_LAYER_NEXT_EXE if(this->Front) this->Front->exe()
	#define SX_MC_LAYER_NEXT_RESET if(this->Front) this->Front->reset()
	#define SX_MC_LAYER_NEXT_FIT if(this->Back) this->Back->fit(nullptr, _Rate, _ErrParam)
	#define SX_MC_LAYER_NEXT_APPLY if(this->Front) this->Front->apply(_Rate)
	#define SX_MC_LAYER_NEXT_STORE if(this->Front) this->Front->store(_Stream)
	#define SX_MC_LAYER_NEXT_LOAD if(this->Front) this->Front->load(_Stream)


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Constants.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	constexpr auto ALIGNMENT = u64(32);


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

		virtual auto outSz ( void ) const -> u64 { return 0; }
		virtual auto outSzBt ( void ) const -> u64 { return this->outSz() * sizeof(T); }
		virtual auto in ( void ) const -> const T* { return this->Input; }
		virtual auto out ( void ) const -> const T* { return nullptr; }
		virtual auto gradient ( void ) const -> const T* { return nullptr; }
		virtual auto back ( void ) -> Layer* { return this->Back; }
		virtual auto front ( void ) -> Layer* { return this->Front; }
		virtual auto setBack ( Layer* _Back ) -> void { this->Back = _Back; if(_Back) { this->Input = _Back->out(); _Back->setFront(this); } }
		virtual auto setFront ( Layer* _Front ) -> void { this->Front = _Front; }

		virtual auto lock ( void ) -> void { this->IsLocked = true; }
		virtual auto unlock ( void ) -> void { this->IsLocked = false; }

		virtual SX_FNSIG_LAYER_EXE = 0;
		virtual SX_FNSIG_LAYER_RESET { if(this->Front) this->Front->reset(); }
		virtual SX_FNSIG_LAYER_ERR { return 0; }
		virtual SX_FNSIG_LAYER_FIT = 0;
		virtual SX_FNSIG_LAYER_APPLY { if(this->Front) this->Front->apply(_Rate); }
		virtual SX_FNSIG_LAYER_STORE { if(this->Front) this->Front->store(_Stream); }
		virtual SX_FNSIG_LAYER_LOAD { if(this->Front) this->Front->load(_Stream); }

		inline auto setInput ( const T* _Input ) -> const T* { const auto InputLast = this->Input; if(_Input) this->Input = _Input; return InputLast; }
	};
}
