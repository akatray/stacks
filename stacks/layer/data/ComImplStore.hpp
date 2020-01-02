		SX_FNSIG_LAYER_STORE final
		{
			_Stream.write(reinterpret_cast<const char*>(this->Weights), SZ_BUF_W * sizeof(T));
			_Stream.write(reinterpret_cast<const char*>(this->Biases), SZ_BUF_B * sizeof(T));

			SX_MC_LAYER_NEXT_STORE;
		}