		SX_FNSIG_LAYER_APPLY final
		{
			const auto Rate = _Rate;
			this->Iter++;

			//if(this->Iter >= 128)
			//{
			//	this->Iter = 0;
			//	if constexpr(needBufM<T,FN_OPTIM>()) memZero(SZ_BUF_W, this->WeightsDltM);
			//	if constexpr(needBufV<T,FN_OPTIM>()) memZero(SZ_BUF_W, this->WeightsDltV);
			//}
			
			if(!this->IsLocked)
			{
				if constexpr(!needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(Rate, this->Iter, SZ_BUF_W, this->Weights, this->WeightsDlt, nullptr, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(Rate, this->Iter, SZ_BUF_W, this->Weights, this->WeightsDlt, this->WeightsDltM, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(Rate, this->Iter, SZ_BUF_W, this->Weights, this->WeightsDlt, this->WeightsDltM, this->WeightsDltV);
				
				if constexpr(!needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(Rate, this->Iter, SZ_BUF_B, this->Biases, this->BiasesDlt, nullptr, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && !needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(Rate, this->Iter, SZ_BUF_B, this->Biases, this->BiasesDlt, this->BiasesDltM, nullptr);
				if constexpr(needBufM<T,FN_OPTIM>() && needBufV<T,FN_OPTIM>()) optimApply<T,FN_OPTIM>(Rate, this->Iter, SZ_BUF_B, this->Biases, this->BiasesDlt, this->BiasesDltM, this->BiasesDltV);
			}

			SX_MC_LAYER_NEXT_APPLY;
		}