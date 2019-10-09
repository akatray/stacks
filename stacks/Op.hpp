// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <fx/Types.hpp>
#include <fx/Error.hpp>
#include <fx/Utilities.hpp>
#include <iostream>

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
	// Stack operation interface.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class Op
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		protected:
		Op* Back;
		Op* Front;
		const utl::Shape ShpIn;
		const utl::Shape ShpOut;
		const r32* Input;
		bool IsLocked;
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Operation utilities.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Tranformation functions.
		enum class Func
		{
			SIGMOID,
			TANH,
			RELU,
			PRELU
		};


		constexpr static auto FUNC_SIGMOID = u64(1);
		constexpr static auto FUNC_TANH = u64(2);
		constexpr static auto FUNC_RELU = u64(3);
		constexpr static auto FUNC_PRELU = u64(4);

		// Error codes.
		constexpr static auto ERR_BAD_SHAPE = u64(1);
		constexpr static auto ERR_LOAD = u64(2);

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Op ( void ) : Front(nullptr), Back(nullptr), ShpIn(), ShpOut(), Input(nullptr), IsLocked(false) {}
		Op ( const utl::Shape& _ShpIn, const utl::Shape& _ShpOut ) : Front(nullptr), Back(nullptr), ShpIn(_ShpIn), ShpOut(_ShpOut), Input(nullptr), IsLocked(false) {}
		virtual ~Op ( void ) {}

		virtual auto id ( void ) const -> u64 { return 0; }
		constexpr inline auto shpin ( void ) const -> const utl::Shape& { return this->ShpIn; }
		constexpr inline auto shpout ( void ) const -> const utl::Shape& { return this->ShpOut; }
		inline auto input ( void ) const -> const r32* { return this->Input; }
		virtual auto output ( void ) -> r32* { return nullptr; }
		virtual auto gradient ( void ) const -> const r32* { return nullptr; }
		virtual auto back ( void ) -> Op* { return this->Back; }
		virtual auto front ( void ) -> Op* { return this->Front; }
		virtual auto setBack ( Op* _Back ) -> void { this->Back = _Back; if(_Back) { this->Input = _Back->output(); _Back->setFront(this); } }
		virtual auto setFront ( Op* _Front ) -> void { this->Front = _Front; }

		virtual auto lock ( void ) -> void { this->IsLocked = true; }
		virtual auto unlock ( void ) -> void { this->IsLocked = false; }

		virtual auto execute ( const r32* _Input ) -> r32* { return nullptr; }
		virtual auto reset ( void ) -> void { return; }
		virtual auto fit ( const r32* _Target, const r32* _Mask = nullptr ) -> void { return; }
		virtual auto apply ( const r32 _Rate = 0.01f ) -> void { return; }
		virtual auto store ( std::ostream& _Stream ) const -> void { return; }
		virtual auto load ( std::istream& _Stream ) -> void { return; }
	};
}