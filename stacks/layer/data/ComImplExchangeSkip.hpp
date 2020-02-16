		// This function expects _Master to be in identical configuration as this.
		// Version for layers without parameters.
		SX_FNSIG_LAYER_EXCHANGE final
		{
			if(this->Front && _Chain) this->Front->exchange(static_cast<decltype(this)>(_Master)->Front);
		}
