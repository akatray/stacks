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
	// Output buffers.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	// Default.
	template<class T, uMAX SZ_OUT, uMAX SZ_GRAD, FnTrans FN_TRANS>
	struct LDOutputs
	{
		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_GRAD];

		LDOutputs ( void ) : OutTrans{}, Gradient{} {}
	};


	// Specialization for relu.
	template<class T, uMAX SZ_OUT, uMAX SZ_GRAD>
	struct LDOutputs<T, SZ_OUT, SZ_GRAD, FnTrans::RELU>
	{
		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T OutRaw[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_GRAD];

		LDOutputs ( void ) : OutTrans{}, OutRaw{}, Gradient{} {}
	};


	// Specialization for prelu.
	template<class T, uMAX SZ_OUT, uMAX SZ_GRAD>
	struct LDOutputs<T, SZ_OUT, SZ_GRAD, FnTrans::PRELU>
	{
		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T OutRaw[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_GRAD];

		LDOutputs ( void ) : OutTrans{}, OutRaw{}, Gradient{} {}
	};

	// Specialization for elu.
	template<class T, uMAX SZ_OUT, uMAX SZ_GRAD>
	struct LDOutputs<T, SZ_OUT, SZ_GRAD, FnTrans::ELU>
	{
		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T OutRaw[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_GRAD];

		LDOutputs ( void ) : OutTrans{}, OutRaw{}, Gradient{} {}
	};
}
