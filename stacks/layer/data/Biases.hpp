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
	// Layer data: Biases buffers.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, uMAX SIZE, bool HAS_BUF_M = true, bool HAS_BUF_V = true>
	struct LDBiases
	{
		alignas(ALIGNMENT) T Biases[SIZE];
		alignas(ALIGNMENT) T BiasesDlt[SIZE];
		alignas(ALIGNMENT) T BiasesDltM[SIZE];
		alignas(ALIGNMENT) T BiasesDltV[SIZE];

		LDBiases ( void ) : Biases{}, BiasesDlt{}, BiasesDltM{}, BiasesDltV{} { rng::rbuf(SIZE, this->Biases, 0.0001, 0.001); }
	};

	template<class T, uMAX SIZE>
	struct LDBiases<T, SIZE, false, false>
	{
		alignas(ALIGNMENT) T Biases[SIZE];
		alignas(ALIGNMENT) T BiasesDlt[SIZE];

		LDBiases ( void ) : Biases{}, BiasesDlt{} { rng::rbuf(SIZE, this->Biases, 0.0001, 0.001); }
	};

	template<class T, uMAX SIZE>
	struct LDBiases<T, SIZE, true, false>
	{
		alignas(ALIGNMENT) T Biases[SIZE];
		alignas(ALIGNMENT) T BiasesDlt[SIZE];
		alignas(ALIGNMENT) T BiasesDltM[SIZE];

		LDBiases ( void ) : Biases{}, BiasesDlt{}, BiasesDltM{} { rng::rbuf(SIZE, this->Biases, 0.0001, 0.001); }
	};

	template<class T, uMAX SIZE>
	struct LDBiases<T, SIZE, false, true>
	{
		alignas(ALIGNMENT) T Biases[SIZE];
		alignas(ALIGNMENT) T BiasesDlt[SIZE];
		alignas(ALIGNMENT) T BiasesDltV[SIZE];

		LDBiases ( void ) : Biases{}, BiasesDlt{}, BiasesDltV{} { rng::rbuf(SIZE, this->Biases, 0.0001, 0.001); }
	};
}
