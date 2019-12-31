// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <fx/Types.hpp>

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
	// Vector operations.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	namespace vops
	{
		template<class T> constexpr inline auto mulOutByConst ( const u64 _Size, T* _Out, const T _Const ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] *= _Const;
		}

		template<class T> constexpr inline auto divOutByConst ( const u64 _Size, T* _Out, const T _Const ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] /= _Const;
		}
		
		template<class T> constexpr inline auto mulVecByConst ( const u64 _Size, T* _Out, const T* _Vec, const T _Const ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] = _Vec[i] * _Const;
		}

		template<class T> constexpr inline auto mulVecByConstAddToOut ( const u64 _Size, T* _Out, const T* _Vec, const T _Const ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] += (_Vec[i] * _Const);
		}

		template<class T> constexpr inline auto mulVecByConstSubFromOut ( const u64 _Size, T* _Out, const T* _Vec, const T _Const ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] -= (_Vec[i] * _Const);
		}

		template<class T> constexpr inline auto mulOutByVec ( const u64 _Size, T* _Out, const T* _Vec ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] *= _Vec[i];
		}

		template<class T> constexpr inline auto mulVecByVec ( const u64 _Size, T* _Out, const T* _Vec0, const T* _Vec1 ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] = _Vec0[i] * _Vec1[i];
		}

		template<class T> constexpr inline auto addVecToOut ( const u64 _Size, T* _Out, const T* _Vec ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] += _Vec[i];
		}
		
		template<class T> constexpr inline auto addVecToVec ( const u64 _Size, T* _Out, const T* _Vec0, const T* _Vec1 ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] = _Vec0[i] + _Vec1[i];
		}

		template<class T> constexpr inline auto subConstFromOut ( const u64 _Size, T* _Out, const T _Const ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] -= _Const;
		}

		template<class T> constexpr inline auto subVecFromOut ( const u64 _Size, T* _Out, const T* _Vec ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] -= _Vec[i];
		}
		
		template<class T> constexpr inline auto subVecFromVec ( const u64 _Size, T* _Out, const T* _Vec0, const T* _Vec1 ) -> void
		{
			for(auto i = u64(0); i < _Size; ++i) _Out[i] = _Vec0[i] - _Vec1[i];
		}

		template<class T> constexpr inline auto mulVecByVecSum ( const u64 _Size, const T* _Vec0, const T* _Vec1 ) -> T
		{
			auto Sum = T(0.0);
			for(auto i = u64(0); i < _Size; ++i) Sum += _Vec0[i] * _Vec1[i];
			return Sum;
		}
	}
}
