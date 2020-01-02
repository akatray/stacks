		SX_FNSIG_LAYER_APPLY final
		{
			if(!this->IsLocked)
			{
				auto PtrWDM = (T*)nullptr;
				auto PtrWDV = (T*)nullptr;
				auto PtrBDM = (T*)nullptr;
				auto PtrBDV = (T*)nullptr;

				if constexpr(needBufM<T,FN_OPTIM>())
				{
					PtrWDM = this->WeightsDltM;
					PtrBDM = this->BiasesDltM;
				}

				if constexpr(needBufV<T,FN_OPTIM>())
				{
					PtrWDV = this->WeightsDltV;
					PtrBDV = this->BiasesDltV;
				}


				optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_W, this->Weights, this->WeightsDlt, PtrWDM, PtrWDV);
				optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_B, this->Biases, this->BiasesDlt, PtrBDM, PtrBDV);
			}

			SX_MC_LAYER_NEXT_APPLY;
		}