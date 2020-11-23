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
	// Initialisation options.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	enum class FnInitWeights
	{
		DEFAULT,

		NRM_SIGMOID,
		NRM_TANH,
		NRM_RELU,

		UNI_SIGMOID,
		UNI_TANH,
		UNI_RELU
	};


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Weights m buffer.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, uMAX SIZE> struct LDWeightsM
	{
		alignas(ALIGNMENT) T WeightsDltM[SIZE];
		LDWeightsM ( void ) : WeightsDltM{}{}
	};
	

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Weights m and v buffers.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, uMAX SIZE> struct LDWeightsMV
	{
		alignas(ALIGNMENT) T WeightsDltM[SIZE];
		alignas(ALIGNMENT) T WeightsDltV[SIZE];
		LDWeightsMV ( void ) : WeightsDltM{}, WeightsDltV{}{}
	};


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Weights and delta buffers.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, FnOptim FN_OPTIM, uMAX SZ_BUF, uMAX SZ_IN, uMAX SZ_OUT, FnInitWeights FN_INIT_W = FnInitWeights::DEFAULT>
	struct LDWeights:
	std::conditional_t<FN_OPTIM == FnOptim::MOMENTUM, LDWeightsM<T, SZ_BUF>, None1>,
	std::conditional_t<FN_OPTIM == FnOptim::ADAM, LDWeightsMV<T, SZ_BUF>, None2>
	{
		alignas(ALIGNMENT) uMAX Iter;
		
		alignas(ALIGNMENT) T Weights[SZ_BUF];
		alignas(ALIGNMENT) T WeightsDlt[SZ_BUF];

		LDWeights ( void ) : Iter(0), Weights{}, WeightsDlt{}
		{
			if constexpr(FN_INIT_W == FnInitWeights::DEFAULT)
			{
				rng::rbuf(SZ_BUF, this->Weights, T(-0.01), T(0.01));
			}

			if constexpr(FN_INIT_W == FnInitWeights::NRM_SIGMOID)
			{
				const auto Variance = T(4) * std::sqrt(T(2) / (SZ_IN + SZ_OUT));
				rng::rbuf_nrm(SZ_BUF, this->Weights, T(0), Variance);
			}

			if constexpr(FN_INIT_W == FnInitWeights::NRM_TANH)
			{
				const auto Variance = std::sqrt(T(2) / (SZ_IN + SZ_OUT));
				rng::rbuf_nrm(SZ_BUF, this->Weights, T(0), Variance);
			}

			if constexpr(FN_INIT_W == FnInitWeights::NRM_RELU)
			{
				const auto Variance =  std::sqrt(T(2)) * std::sqrt(T(2) / (SZ_IN + SZ_OUT));
				rng::rbuf_nrm(SZ_BUF, this->Weights, T(0), Variance);
			}

			if constexpr(FN_INIT_W == FnInitWeights::UNI_SIGMOID)
			{
				const auto Range = T(4) * std::sqrt(T(6) / (SZ_IN + SZ_OUT));
				rng::rbuf(SZ_BUF, this->Weights, -Range, Range);
			}

			if constexpr(FN_INIT_W == FnInitWeights::UNI_TANH)
			{
				const auto Range = std::sqrt(T(6) / (SZ_IN + SZ_OUT));
				rng::rbuf(SZ_BUF, this->Weights, -Range, Range);
			}

			if constexpr(FN_INIT_W == FnInitWeights::UNI_RELU)
			{
				const auto Range = std::sqrt(T(2)) * std::sqrt(T(6) / (SZ_IN + SZ_OUT));
				rng::rbuf(SZ_BUF, this->Weights, -Range, Range);
			}
		}
	};
}
