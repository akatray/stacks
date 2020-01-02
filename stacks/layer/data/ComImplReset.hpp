		SX_FNSIG_LAYER_RESET final
		{
			if(!this->IsLocked)
			{
				memZero(SZ_BUF_W, this->WeightsDlt);
				memZero(SZ_BUF_B, this->BiasesDlt);
			}
			
			SX_MC_LAYER_NEXT_RESET;
		}