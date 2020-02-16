		// This function expects _Master to be in identical configuration as this.
		SX_FNSIG_LAYER_EXCHANGE final
		{
			auto Master = static_cast<decltype(this)>(_Master);

			memCopy(SZ_BUF_W, Master->WeightsDlt, this->WeightsDlt);
			memCopy(SZ_BUF_B, Master->BiasesDlt, this->BiasesDlt);
			memCopy(SZ_BUF_W, this->Weights, Master->Weights);
			memCopy(SZ_BUF_B, this->Biases, Master->Biases);

			if(this->Front && _Chain) this->Front->exchange(Master->Front);
		}
