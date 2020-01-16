		SX_FNSIG_LAYER_APPLY final
		{
			if(!this->IsLocked)
			{
				if constexpr(!needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_W, this->Weights, this->WeightsDlt, nullptr, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_W, this->Weights, this->WeightsDlt, this->WeightsDltM, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_W, this->Weights, this->WeightsDlt, this->WeightsDltM, this->WeightsDltV);
				
				if constexpr(!needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_B, this->Biases, this->BiasesDlt, nullptr, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_B, this->Biases, this->BiasesDlt, this->BiasesDltM, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(_Rate, SZ_BUF_B, this->Biases, this->BiasesDlt, this->BiasesDltM, this->BiasesDltV);
			}

			SX_MC_LAYER_NEXT_APPLY;
		}