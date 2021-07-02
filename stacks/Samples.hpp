// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once


// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <fx/Types.hpp>
#include <fx/Image.hpp>
#include <fx/Files.hpp>


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
	// Load samples cache.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> auto loadSamplesCache ( const str& _CacheFile )
	{
		// Open cache file.
		auto CacheFile = std::ifstream(_CacheFile, std::ios::binary);
		if(CacheFile.is_open())
		{
			// Read metadata.
			auto SamplesCount = u64(0);
			auto SampleSize = u64(0);

			CacheFile.seekg(-(i64(sizeof(u64)*2)), CacheFile.end);
			CacheFile.read(reinterpret_cast<char*>(&SamplesCount), sizeof(u64));
			CacheFile.read(reinterpret_cast<char*>(&SampleSize), sizeof(u64));

			CacheFile.seekg(0, CacheFile.beg);


			// Load samples.
			auto Samples = vec<vec<T>>();
			for(auto s = uMAX(0); s < SamplesCount; ++s)
			{
				Samples.push_back(vec<T>(SampleSize / sizeof(T)));
				CacheFile.read(reinterpret_cast<char*>(Samples.back().data()), SampleSize);
				std::cout << "Loading cache file [" << _CacheFile << "][" << s+1 << "/" << SamplesCount << "].\n";
			}


			// Finish.
			return std::move(Samples);
		}
	}


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Store samples cache.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> auto storeSamplesCache ( const str& _CacheFile, const vec<vec<T>>& _Samples )
	{
		// Open cache file.
		auto CacheFile = std::ofstream(_CacheFile, std::ios::binary);
		if(CacheFile.is_open())
		{
			// Metadata.
			const auto SamplesCount = u64(_Samples.size());
			const auto SampleSize = u64(_Samples[0].size() * sizeof(T));

			// Store samples.
			for(auto s = uMAX(0); s < SamplesCount; ++s)
			{
				CacheFile.write(reinterpret_cast<const char*>(_Samples[s].data()), SampleSize);
				std::cout << "Storing cache file [" << _CacheFile << "][" << s+1 << "/" << SamplesCount << "].\n";
			}

			// Write metadata.
			CacheFile.write(reinterpret_cast<const char*>(&SamplesCount), sizeof(u64));
			CacheFile.write(reinterpret_cast<const char*>(&SampleSize), sizeof(u64));
		}
	}


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Configuration to load images as samples.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	struct CfgBuildImgChc
	{
		uMAX Width; // Scale image to this width.
		uMAX Height; // Scale image to this height.
		bool SplitChannels; // Splits color channels into seperate images with depth of 1.
		uMAX DownUp; // Downscale by this factor then upscale back.

		CfgBuildImgChc ( const uMAX _Width = 64, const uMAX _Height = 64, const bool _SplitChannels = true, const uMAX _DownUp = 1 ) :
		Width(_Width),
		Height(_Height),
		SplitChannels(_SplitChannels),
		DownUp(_DownUp)
		{}
	};


	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Build samples cache from images.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T> auto buildImageSamplesCache ( const str& _SrcDir, const str& _DstFile, const CfgBuildImgChc& _Cfg ) -> void
	{
		// Open cache file.
		std::cout << "Building samples cache from ["s << _SrcDir << "].\n"s; 
		auto CacheFile = std::ofstream(_DstFile, std::ios::binary);
		

		// Metadata.
		auto SamplesCount = u64(0);
		auto SampleSize = u64(0);

		if(_Cfg.SplitChannels) SampleSize = _Cfg.Width * _Cfg.Height * sizeof(T);
		else SampleSize = _Cfg.Width * _Cfg.Height * 3 * sizeof(T);


		// Process images.
		auto Files = files::buildFileList(_SrcDir, true);
		for(auto& File : Files)
		{
			try
			{
				// Load image.
				std::cout << "Processing ["s << File.string() << "].'\n"s; 
				auto Img = Image<u8>(File.string());


				// Rescale image.
				if((Img.width() != _Cfg.Width) || (Img.height() != _Cfg.Height)) Img = img::resize(Img, _Cfg.Width, _Cfg.Height);
				

				// Degrade image by downscaling and upscaling.
				if(_Cfg.DownUp != 1)
				{
					Img = img::resize(Img, _Cfg.Width / _Cfg.DownUp, _Cfg.Height / _Cfg.DownUp);
					Img = img::resize(Img, _Cfg.Width, _Cfg.Height);
				}


				// Convert to template type.
				auto ImgT = Image<T>(Img); 


				// Split channels. Convolutional layer needs sequential channels.
				auto ImgTC = img::split(ImgT);


				// Write samples to cache.
				for(auto& Channel : ImgTC)
				{
					CacheFile.write(reinterpret_cast<const char*>(Channel.data()), _Cfg.Width * _Cfg.Height * sizeof(T));
					if(_Cfg.SplitChannels) SamplesCount++;
				}

				if(!_Cfg.SplitChannels) SamplesCount++;
			}


			catch(...)
			{
				std::cout << "Failed ["s << File.string() << "]!\n"s;
				continue;
			}
		}


		// Write metadata.
		CacheFile.write(reinterpret_cast<const char*>(&SamplesCount), sizeof(u64));
		CacheFile.write(reinterpret_cast<const char*>(&SampleSize), sizeof(u64));
	}
}
