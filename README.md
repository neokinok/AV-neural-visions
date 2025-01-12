# AV-neural-visions

GENERATIVE VIDEO WITH AI based on real time multiple camera feeds for EXPANDED REALITY

This project aims to record and stream audio and video from multiple ESP32 cameras and include machine learning as seeds for the GAN. 

Image Generation
Generative Adversarial Networks (GANs):
GANs consist of two neural networks, a generator and a discriminator, that compete against each other to produce realistic images.
Popular GAN architectures include DCGAN, StyleGAN, and BigGAN.
Variational Autoencoders (VAEs):
VAEs are another type of generative model that learns to encode and decode data, generating new images that resemble the training data.

Sound Generation
WaveNet:
Developed by DeepMind, WaveNet is a deep generative model for generating raw audio waveforms, and it has been used for text-to-speech (TTS) applications.
Generative Models for Audio Synthesis:

Models like GANs and VAEs can also be adapted for audio generation.
Example Libraries and Frameworks
TensorFlow and PyTorch: Popular deep learning frameworks that provide tools and libraries for building and training generative models.
Keras: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.


Setting up ESP32 cameras for streaming: Configure multiple ESP32 cameras to capture and stream video to a central server or processing unit.

Recording and storing video data: Collect and store video data from the ESP32 cameras for further processing.

Machine learning with GANs: Use the recorded video data as seeds for a Generative Adversarial Network (GAN) to create new content.

### Prerequisites

- ESP32-CAM modules
- Python 3.x
- TensorFlow or PyTorch
- FFmpeg

### Installation

1. **ESP32 Camera Setup**:
   - Flash the `camera_code.ino` to your ESP32-CAM modules.

2. **Server Setup**:
   - Navigate to the `server` directory.
   - Install the required Python packages:
     ```sh
     pip install -r requirements.txt
     ```
   - Run the server to start recording and streaming:
     ```sh
     python app.py
     ```

3. **GAN Training**:
   - Navigate to the `gan_training` directory.
   - Run the GAN training script:
     ```sh
     python train_gan.py
     ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
