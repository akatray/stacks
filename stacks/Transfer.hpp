// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

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
		PRELU
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
	template<class T> constexpr inline auto relu (const T _X) { return std::max(T(0), _X); }
	template<class T> constexpr inline auto reluDer (const T _X) { if(_X >= T(0)) return T(1); else return T(0); }

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Parametric REctified Linear Unit.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> constexpr inline auto prelu (const T _X, const T _A = T(0.2)) { if(_X >= T(0)) return _X; else return _X * _A; }
	template<class T> constexpr inline auto preluDer (const T _X, const T _A = T(0.2)) { if(_X >= T(0)) return T(1); else return _A; }

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Does transfer function needs pre-transfer value.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, FnTrans FN_TRANS> constexpr inline auto needRaw ( void )
	{
		if constexpr((FN_TRANS == FnTrans::RELU) || (FN_TRANS == FnTrans::PRELU)) return true;
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
		if constexpr(FN_TRANS == FnTrans::PRELU) return prelu(_Val, T(0.01));
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Transfer derivative.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, FnTrans FN_TRANS> constexpr inline auto transferDer ( const T _TVal, const T _RVal )
	{
		if constexpr(FN_TRANS == FnTrans::LINEAR) return T(1.0);
		if constexpr(FN_TRANS == FnTrans::SIGMOID) return sigmoidDer2(_TVal);
		if constexpr(FN_TRANS == FnTrans::TANH) return tanhDer2(_TVal);
		if constexpr(FN_TRANS == FnTrans::RELU) return reluDer(_RVal);
		if constexpr(FN_TRANS == FnTrans::PRELU) return preluDer(_RVal, T(0.01));
	}
}
