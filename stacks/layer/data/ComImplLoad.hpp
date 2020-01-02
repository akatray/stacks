		SX_FNSIG_LAYER_LOAD final
		{
			_Stream.read(reinterpret_cast<char*>(this->Weights), SZ_BUF_W * sizeof(T));
			_Stream.read(reinterpret_cast<char*>(this->Biases), SZ_BUF_B * sizeof(T));

			SX_MC_LAYER_NEXT_LOAD;
		}