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
	// Layer data: Weights buffers.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, uMAX SZ_BUF, uMAX SZ_IN, uMAX SZ_OUT, bool HAS_BUF_M = true, bool HAS_BUF_V = true>
	struct LDWeights
	{
		alignas(ALIGNMENT) T Weights[SZ_BUF];
		alignas(ALIGNMENT) T WeightsDlt[SZ_BUF];
		alignas(ALIGNMENT) T WeightsDltM[SZ_BUF];
		alignas(ALIGNMENT) T WeightsDltV[SZ_BUF];

		LDWeights ( void ) : Weights{}, WeightsDlt{}, WeightsDltM{}, WeightsDltV{} { rng::rbuf(SZ_BUF, this->Weights, -std::sqrt(T(6) / (T(SZ_IN), T(SZ_OUT))), std::sqrt(T(6) / (T(SZ_IN), T(SZ_OUT)))); }
	};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	template<class T, uMAX SZ_BUF, uMAX SZ_IN, uMAX SZ_OUT>
	struct LDWeights<T, SZ_BUF, SZ_IN, SZ_OUT, false, false>
	{
		alignas(ALIGNMENT) T Weights[SZ_BUF];
		alignas(ALIGNMENT) T WeightsDlt[SZ_BUF];

		LDWeights ( void ) : Weights{}, WeightsDlt{} { rng::rbuf(SZ_BUF, this->Weights, -std::sqrt(T(6) / (T(SZ_IN), T(SZ_OUT))), std::sqrt(T(6) / (T(SZ_IN), T(SZ_OUT)))); }
	};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	template<class T, uMAX SZ_BUF, uMAX SZ_IN, uMAX SZ_OUT>
	struct LDWeights<T, SZ_BUF, SZ_IN, SZ_OUT, true, false>
	{
		alignas(ALIGNMENT) T Weights[SZ_BUF];
		alignas(ALIGNMENT) T WeightsDlt[SZ_BUF];
		alignas(ALIGNMENT) T WeightsDltM[SZ_BUF];

		LDWeights ( void ) : Weights{}, WeightsDlt{}, WeightsDltM{} { rng::rbuf(SZ_BUF, this->Weights, -std::sqrt(T(6) / (T(SZ_IN), T(SZ_OUT))), std::sqrt(T(6) / (T(SZ_IN), T(SZ_OUT)))); }
	};
}
