// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once


// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <algorithm>


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
	// Transfer function options.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	enum class FnTrans
	{
		LINEAR,
		SIGMOID,
		TANH,
		RELU,
		PRELU,
		ELU,
		GELU
	};


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Logistic / sigmoid.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> inline auto sigmoid (const T _X) { return (T(1) / (T(1) + std::exp(-_X))); }
	template<class T> inline auto sigmoidDer (const T _X) { return (sigmoid(_X) * (T(1) - sigmoid(_X))); }
	template<class T> constexpr inline auto sigmoidDer2 (const T _FX) { return (_FX * (T(1) - _FX)); }


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// TanH.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> inline auto tanh (const T _X) { return (T(2) / (T(1) + std::exp(-_X * T(2)))) - T(1); }
	template<class T> inline auto tanhDer2 (const T _FX) { return T(1) - std::pow(_FX, T(2)); }


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// REctified Linear Unit.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//template<class T> constexpr inline auto relu ( const T _X, const T _Floor = T(-1) ) { return std::max(_Floor, _X); }
	//template<class T> constexpr inline auto reluDer ( const T _X, const T _Floor = T(-1) ) { return _X > _Floor; }

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> constexpr inline auto relu ( const T _X, const T _Floor = T(-1) ) { if((_X > T(-1)) && (_X < T(1))) return _X; else return T(0); }
	template<class T> constexpr inline auto reluDer ( const T _X, const T _Floor = T(-1) ) { if((_X > T(-1)) && (_X < T(1))) return T(1); else return T(0); }


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Parametric REctified Linear Unit.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> constexpr inline auto prelu (const T _X, const T _A = T(0.1)) { if(_X > T(-1)) return _X; else return _X * _A; }
	template<class T> constexpr inline auto preluDer (const T _X, const T _A = T(0.1)) { if(_X > T(-1)) return T(1); else return _A; }


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Exponential Linear Unit.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> inline auto elu (const T _X, const T _A = T(0.01)) { if(_X > T(-1)) return _X; else return _A * (std::exp(_X)-T(1)); }
	template<class T> inline auto eluDer (const T _X, const T _A = T(0.01)) { if(_X > T(-1)) return T(1); else return _A * std::exp(_X); }

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Gaussian Error Linear Unit.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> inline auto sech ( const T _X ) { return (T(2) * std::exp(_X)) / (std::exp(_X * T(2)) + T(1)); }
	/*
	template<class T> inline auto elu ( const T _X ) { return T(0.5) * _X * (T(1) + tanh(T(0.797884) * (_X + T(0.044715) * std::pow(_X, T(3))))); }
	template<class T> inline auto eluDer ( const T _X )
	{
		return T(0.5) * tanh(T(0.0356774) * std::pow(_X, T(3)) + T(0.797884) * _X) +
		(T(0.0535161) * std::pow(_X, T(3)) + T(0.398942) * _X) * std::pow(sech((0.0356774 * std::pow(_X, T(3)) + T(0.797884) * _X)), T(2)) + T(0.5);
		}
		//*/

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Does transfer function needs pre-transfer value.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, FnTrans FN_TRANS> constexpr inline auto needRaw ( void )
	{
		if constexpr((FN_TRANS == FnTrans::RELU) || (FN_TRANS == FnTrans::PRELU) || (FN_TRANS == FnTrans::ELU)) return true;
		else return false;
	}


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Transfer apply.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, FnTrans FN_TRANS> constexpr inline auto transfer ( const T _Val )
	{
		if constexpr(FN_TRANS == FnTrans::LINEAR) return _Val;
		if constexpr(FN_TRANS == FnTrans::SIGMOID) return sigmoid(_Val);
		if constexpr(FN_TRANS == FnTrans::TANH) return tanh(_Val);
		if constexpr(FN_TRANS == FnTrans::RELU) return relu(_Val);
		if constexpr(FN_TRANS == FnTrans::PRELU) return prelu(_Val);
		if constexpr(FN_TRANS == FnTrans::ELU) return elu(_Val);
	}


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Transfer derivative.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, FnTrans FN_TRANS> constexpr inline auto transferDer ( const T _Val )
	{
		if constexpr(FN_TRANS == FnTrans::LINEAR) return T(1);
		if constexpr(FN_TRANS == FnTrans::SIGMOID) return sigmoidDer2(_Val);
		if constexpr(FN_TRANS == FnTrans::TANH) return tanhDer2(_Val);
		if constexpr(FN_TRANS == FnTrans::RELU) return reluDer(_Val);
		if constexpr(FN_TRANS == FnTrans::PRELU) return preluDer(_Val);
		if constexpr(FN_TRANS == FnTrans::ELU) return eluDer(_Val);
	}


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Sigmoid transfer function.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> struct FnTransSigmoid
	{
		constexpr static auto RAW = false;

		constexpr static inline T trans ( const T _X )
		{ 
			return (T(1) / (T(1) + std::exp(-_X)));
		}
		
		constexpr static inline T der ( const T _FX )
		{
			return (_FX * (T(1) - _FX));
		}
	};

	template<class T> using FnTrSigmoid = sx::FnTransSigmoid<T>;


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Tanh transfer function.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> struct FnTransTanh
	{
		constexpr static auto RAW = false;

		constexpr static inline T trans ( const T _X )
		{ 
			return (T(2) / (T(1) + std::exp(-_X * T(2)))) - T(1);
		}
		
		constexpr static inline T der ( const T _FX )
		{
			return T(1) - std::pow(_FX, T(2));
		}
	};

	template<class T> using FnTrTanh = sx::FnTransTanh<T>;


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// RELU transfer function.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, iMAX FLOOR, iMAX CEIL> struct FnTransRelu
	{
		constexpr static auto RAW = true;

		constexpr static inline T trans ( const T _X )
		{ 
			if constexpr((FLOOR == 0) && (CEIL == 0)) return std::max(T(0), _X);
			if constexpr((FLOOR != 0) && (CEIL == 0)) return std::max(T(FLOOR), _X);
			if constexpr((FLOOR != 0) && (CEIL != 0)) { if(T(FLOOR) < _X < T(CEIL)) return _X; else return T(0); }
		}
		
		constexpr static inline T der ( const T _X )
		{
			if constexpr((FLOOR == 0) && (CEIL == 0)) return _X > T(0);
			if constexpr((FLOOR != 0) && (CEIL == 0)) return _X > T(FLOOR);
			if constexpr((FLOOR != 0) && (CEIL != 0)) { if(T(FLOOR) < _X < T(CEIL)) return T(1); else return T(0); }
		}
	};

	template<class T> using FnTrRelu = sx::FnTransRelu<T,0,0>;
	template<class T> using FnTrReluNeg = sx::FnTransRelu<T,-1,0>;
	template<class T> using FnTrReluCon = sx::FnTransRelu<T,-2,2>;
}
