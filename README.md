# AV-neural-visions
This project aims to record and stream audio and video from multiple ESP32 cameras and include machine learning as seeds for the GAN. 
## Getting Started

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
