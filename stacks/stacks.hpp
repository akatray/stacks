// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Constants.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace sx
{
	constexpr auto VERSION_MAJOR = int(6);
	constexpr auto VERSION_MINOR = int(0);
	constexpr auto VERSION_PATCH = int(0);
	constexpr auto ALIGNMENT = int(32);
}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Nothing.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	struct None1 {};
	struct None2 {};
	struct None3 {};
	struct None4 {};
	struct None5 {};


// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <fx/Types.hpp>
#include <fx/Rng.hpp>
#include <fx/Math.hpp>
#include <fx/Vops.hpp>

#include "./Samples.hpp"

#include "./Network.hpp"

#include "./Layer.hpp"

#include "./layer/Error.hpp"
#include "./layer/ErrorConv2.hpp"

#include "./layer/Filter.hpp"

#include "./layer/Dense.hpp"

#include "./layer/Downscale2.hpp"
#include "./layer/Upscale2.hpp"
#include "./layer/Conv2.hpp"
