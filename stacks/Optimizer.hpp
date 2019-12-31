// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <fx/Types.hpp>
#include "fn_vector.hpp"

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
	// Constants.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	constexpr auto BETA1 = r64(0.9);
	constexpr auto BETA1F = r64(1.0) - BETA1;
	constexpr auto BETA2 = r64(0.99999999);
	constexpr auto BETA2F = r64(1.0) - BETA2;
	constexpr auto EPSILON = r64(1e-8);

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Optimizer options.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	enum class FnOptim
	{
		NONE,
		MOMENTUM,
		ADAM,
	};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Optimized apply.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template
	<
		class T,
		FnOptim FN_OPTIM
	>
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	auto optimApply
	(
		const T _Rate,
		
		const u64 _Size,
		T* _Buff,
		T* _BuffD,
		T* _BuffM,
		T* _BuffV

	)
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	{
		if constexpr(FN_OPTIM == FnOptim::NONE) vops::mulVecByConstSubFromOut(_Size, _Buff, _BuffD, _Rate);
			

		if constexpr(FN_OPTIM == FnOptim::MOMENTUM)
		{
			for(auto i = u64(0); i < _Size; ++i)
			{
				_BuffM[i] = (_BuffM[i] * BETA1) + (_BuffD[i] * BETA1F);
				_Buff[i] -= _Rate * _BuffM[i];
			}
		}


		if constexpr(FN_OPTIM == FnOptim::ADAM)
		{
			for(auto i = u64(0); i < _Size; ++i)
			{
				_BuffM[i] = (_BuffM[i] * BETA1) + (_BuffD[i] * BETA1F);
				_BuffV[i] = (_BuffV[i] * BETA2) + ((_BuffD[i] * _BuffD[i]) * BETA2F);
				_Buff[i] -= _Rate * ((_BuffM[i] / BETA1F) / (std::sqrt(_BuffV[i] / BETA2F) + EPSILON));
			}
		}
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
}
