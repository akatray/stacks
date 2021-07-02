		SX_FNSIG_LAYER_RESET final
		{
			if(!this->IsLocked)
			{
				memZero(SZ_BUF_W, this->WeightsDlt);

				//for(auto i = uMAX(0); i < SZ_BUF_W; ++i)
				//{
				//	this->WeightsDlt[i] = T(1);
				//}
//
				memZero(SZ_BUF_B, this->BiasesDlt);
			}
			
			SX_MC_LAYER_NEXT_RESET;
		}