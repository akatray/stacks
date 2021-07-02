// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once


// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Layer outputs.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	// Default.
	template<class T, int SZ_OUT, int SZ_GRAD, bool RAW = false>
	struct Outputs_ld
	{
		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_GRAD];

		Outputs_ld ( void ) : OutTrans{}, Gradient{} {}
	};


	// Default + RAW buffer.
	template<class T, int SZ_OUT, int SZ_GRAD>
	struct Outputs_ld<T, SZ_OUT, SZ_GRAD, true>
	{
		alignas(ALIGNMENT) T OutTrans[SZ_OUT];
		alignas(ALIGNMENT) T OutRaw[SZ_OUT];
		alignas(ALIGNMENT) T Gradient[SZ_GRAD];

		Outputs_ld ( void ) : OutTrans{}, OutRaw{}, Gradient{} {}
	};
}
