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
	template<class T, uMAX SIZE, bool HAS_BUF_M = true, bool HAS_BUF_V = true>
	struct LDWeights
	{
		alignas(ALIGNMENT) T Weights[SIZE];
		alignas(ALIGNMENT) T WeightsDlt[SIZE];
		alignas(ALIGNMENT) T WeightsDltM[SIZE];
		alignas(ALIGNMENT) T WeightsDltV[SIZE];

		LDWeights ( void ) : Weights{}, WeightsDlt{}, WeightsDltM{}, WeightsDltV{} { rng::rbuf(SIZE, this->Weights, 0.0001, 0.001); }
	};

	template<class T, uMAX SIZE>
	struct LDWeights<T, SIZE, false, false>
	{
		alignas(ALIGNMENT) T Weights[SIZE];
		alignas(ALIGNMENT) T WeightsDlt[SIZE];

		LDWeights ( void ) : Weights{}, WeightsDlt{} { rng::rbuf(SIZE, this->Weights, 0.0001, 0.001); }
	};

	template<class T, uMAX SIZE>
	struct LDWeights<T, SIZE, true, false>
	{
		alignas(ALIGNMENT) T Weights[SIZE];
		alignas(ALIGNMENT) T WeightsDlt[SIZE];
		alignas(ALIGNMENT) T WeightsDltM[SIZE];

		LDWeights ( void ) : Weights{}, WeightsDlt{}, WeightsDltM{} { rng::rbuf(SIZE, this->Weights, 0.0001, 0.001); }
	};

	template<class T, uMAX SIZE>
	struct LDWeights<T, SIZE, false, true>
	{
		alignas(ALIGNMENT) T Weights[SIZE];
		alignas(ALIGNMENT) T WeightsDlt[SIZE];
		alignas(ALIGNMENT) T WeightsDltV[SIZE];

		LDWeights ( void ) : Weights{}, WeightsDlt{}, WeightsDltV{} { rng::rbuf(SIZE, this->Weights, 0.0001, 0.001); }
	};
}
