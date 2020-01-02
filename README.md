## Neural networks implementation.

sx::Network<float> Encode(sx::CompClass::LAYERS);
sx::Network<float> Decode(sx::CompClass::LAYERS);
sx::Network<float> Autoencoder(sx::CompClass::NETWORKS);

Encode.attach(new sx::Dense<float, SAMPLE_SIZE, LATENT_SIZE>());
Encode.attach(new sx::Filter<float, sx::Effect::NORMALIZE, LATENT_SIZE>());

Decode.attach(new sx::Dense<float, LATENT_SIZE, SAMPLE_SIZE>());

Autoencoder.attach(&Encode);
Autoencoder.attach(&Decode);

Autoencoder.loadFromFile("./encoder.file");

Autoencoder.exe(SAMPLE_PTR);
auto Error = Autoencoder.err(SAMPLE_PTR);