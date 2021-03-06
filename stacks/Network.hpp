// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include "./Layer.hpp"
#include <vector>
#include <fstream>

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
	// Default network info structure.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	struct NetworkInfo
	{
		uMAX SxVerMajor;
		uMAX SxVerMinor;
		uMAX SxVerPatch;

		uMAX Epochs;
		uMAX TrainTime;
		uMAX TrainUniqueSamples;

		rMAX ErrMin;
		rMAX ErrMax;
		rMAX ErrAvg;

		NetworkInfo ( void ) :
			SxVerMajor(VERSION_MAJOR),
			SxVerMinor(VERSION_MINOR),
			SxVerPatch(VERSION_PATCH),
			Epochs(0),
			TrainTime(0),
			TrainUniqueSamples(0),
			ErrMin(0),
			ErrMax(0),
			ErrAvg(0)
		{}
	};

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Network components class.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	enum class CompClass
	{
		LAYERS,
		NETWORKS
	};
	
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Neural network.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	template<class T, CompClass MODE> class Network
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		const bool AutoDelete;
		//std::vector<ptr> Components;
		public:
		std::vector<ptr> Components;

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Constructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		Network ( void ) : AutoDelete(true), Components() {}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Destructor.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		~Network ( void )
		{
			if(this->AutoDelete) this->freeLayers();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Attach component.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto attach ( Layer<T>* _Layer ) { if(MODE == CompClass::LAYERS) this->Components.push_back(_Layer); else throw Error("sx"s, "Network<T>"s, "attach"s, 0, "Wrong mode!"s); }
		auto attach ( Network<T,sx::CompClass::LAYERS>* _Model ) { if(MODE == CompClass::NETWORKS) this->Components.push_back(_Model); else throw Error("sx"s, "Network<T>"s, "attach"s, 0, "Wrong mode!"s);}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get most front layer in network.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto front ( void )
		{
			if(MODE == CompClass::LAYERS) return reinterpret_cast<Layer<T>*>(this->Components.front());
			if(MODE == CompClass::NETWORKS) return reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components.front())->front();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Get most back layer in network.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto back ( void )
		{
			if(MODE == CompClass::LAYERS) return reinterpret_cast<Layer<T>*>(this->Components.back());
			if(MODE == CompClass::NETWORKS) return reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components.back())->back();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Setup network for use.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto connect ( void ) -> void
		{
			// Connect layers.
			if(MODE == CompClass::LAYERS)
			{
				for(auto i = uMAX(0); i < this->Components.size(); ++i)
				{
					if(i == 0) reinterpret_cast<Layer<T>*>(this->Components[i])->setBack(nullptr); // First layer has no back layer.
					if(i != 0) reinterpret_cast<Layer<T>*>(this->Components[i])->setBack(reinterpret_cast<Layer<T>*>(this->Components[i-1])); // Chain layers.
					if(i == (this->Components.size() - 1)) reinterpret_cast<Layer<T>*>(this->Components[i])->setFront(nullptr); // Last layer has no front layer.
				}
			}

			// Connect networks.
			if(MODE == CompClass::NETWORKS)
			{
				// Recursively connect subnetworks.
				for(auto i = uMAX(0); i < this->Components.size(); ++i)
				{
					reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components[i])->connect();
				}

				// Connect subnetworks tails.
				for(auto i = uMAX(0); i < this->Components.size(); ++i)
				{
					if(i != 0) reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components[i])->front()->setBack(reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components[i-1])->back());
				}

				// Cut of tails.
				this->front()->setBack(nullptr);
				this->back()->setFront(nullptr);
			}
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Call delete on layer pointers.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto freeLayers ( void ) -> void
		{
			// Delete containing layers.
			if(MODE == CompClass::LAYERS) for(auto i = uMAX(0); i < this->Components.size(); ++i) delete reinterpret_cast<Layer<T>*>(this->Components[i]);

			// Forwards request to subnetworks.
			if(MODE == CompClass::NETWORKS) for(auto i = uMAX(0); i < this->Components.size(); ++i) reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components[i])->freeLayers();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Lock layers from updates.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto lock ( void ) -> void
		{
			// Lock containing layers.
			if(MODE == CompClass::LAYERS) for(auto i = uMAX(0); i < this->Components.size(); ++i) reinterpret_cast<Layer<T>*>(this->Components[i])->lock();

			// Forwards request to subnetworks.
			if(MODE == CompClass::NETWORKS) for(auto i = uMAX(0); i < this->Components.size(); ++i) reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components[i])->lock();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Unlock layers from updates.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto unlock ( void ) -> void
		{
			// Unlock containing layers.
			if(MODE == CompClass::LAYERS) for(auto i = uMAX(0); i < this->Components.size(); ++i) reinterpret_cast<Layer<T>*>(this->Components[i])->unlock();

			// Forwards request to subnetworks.
			if(MODE == CompClass::NETWORKS) for(auto i = uMAX(0); i < this->Components.size(); ++i) reinterpret_cast<Network<T,sx::CompClass::LAYERS>*>(this->Components[i])->unlock();
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Mirror layer functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto outSz ( const bool _Connect = true ) const -> u64 { if(_Connect) this->connect(); return this->back()->outSz(); }
		auto outSzBt ( const bool _Connect = true ) const -> u64 { if(_Connect) this->connect(); return this->back()->outSzBt(); }
		auto in ( const bool _Connect = true ) -> const T* { if(_Connect) this->connect(); return this->front()->in(); }
		auto out ( const bool _Connect = true ) -> const T* { if(_Connect) this->connect(); return this->back()->out(); }
		
		auto exe ( const T* _Input, const bool _Connect = true ) -> void { if(_Connect) this->connect(); this->front()->setInput(_Input); this->front()->exe(); }
		auto reset ( const bool _Connect = true ) -> void { if(_Connect) this->connect(); this->front()->reset(); }
		auto err ( const T* _Target, const bool _Connect = true ) -> T { if(_Connect) this->connect(); return this->back()->err(_Target); }
		auto fit ( const T* _Target, const T _ErrParam, const bool _Connect = true ) -> void { if(_Connect) this->connect(); return this->back()->fit(_Target, _ErrParam); }
		auto apply ( const rMAX _Rate, const uMAX _Iter, const bool _Connect = true ) -> void { if(_Connect) this->connect(); return this->front()->apply(_Rate, _Iter); }


		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store network to file.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto storeToFile ( const std::string& _Filename, const bool _Connect = true )
		{
			if(_Connect) this->connect();


			auto File = std::ofstream(_Filename, std::ios::binary);
			
			if(File.is_open())
			{
				this->front()->store(File);
			}
		}


		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Store network to file with custom data.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		template<class DATA> auto storeToFile ( const std::string& _Filename, const DATA& _Data, const bool _Connect = true )
		{
			if(_Connect) this->connect();


			auto File = std::ofstream(_Filename, std::ios::binary);

			if(File.is_open())
			{
				File.write(reinterpret_cast<const char*>(&_Data), sizeof(DATA));
				this->front()->store(File);
			}
		}


		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load network from file.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto loadFromFile ( const std::string& _Filename, const bool _Connect = true )
		{
			if(_Connect) this->connect();


			auto File = std::ifstream(_Filename, std::ios::binary);

			if(File.is_open())
			{
				this->front()->load(File);
			}
		}


		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Load network from file with custom data.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		template<class DATA> auto loadFromFile ( const std::string& _Filename, DATA& _Data, const bool _Connect = true )
		{
			if(_Connect) this->connect();


			auto File = std::ifstream(_Filename, std::ios::binary);

			if(File.is_open())
			{
				File.read(reinterpret_cast<char*>(&_Data), sizeof(DATA));
				this->front()->load(File);
			}
		}
	};
}
