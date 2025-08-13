# 🚗 Car Damage Assessment System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)](https://nvidia.com)

A professional AI-powered vehicle damage analysis system built with Streamlit and powered by Llama-3.2-11B-Vision model fine-tuned for automotive damage assessment.

## 🎯 Overview

This application provides automated car damage detection and assessment using advanced computer vision and natural language processing. It features a modern web interface, GPU optimization, and professional-grade analysis capabilities.

## ✨ Features

- **🔍 Automated Damage Detection**: AI-powered analysis of vehicle damage from images
- **📊 Severity Classification**: Categorizes damage as minor, moderate, or major
- **🌍 Multi-language Support**: Analysis in French and English
- **📱 Modern UI**: Professional gradient design with responsive layout
- **🚀 GPU Acceleration**: Optimized for NVIDIA GPUs with memory management
- **📸 Multiple Input Methods**: File upload or camera capture
- **⚡ Real-time Processing**: Fast inference with Unsloth optimization
- **🛡️ Robust Error Handling**: Multiple fallback strategies for reliability

## 🏗️ Architecture

### Models
- **Primary**: `Kakyoin03/car-damage-assessment-llama-vision` (Fine-tuned Llama Vision)
- **Fallback**: `KHAOULA-KH/LOra_modele` (LoRA fine-tuned model)
- **Base**: `meta-llama/Llama-3.2-11B-Vision-Instruct` (Foundation model)

### Framework Stack
- **Frontend**: Streamlit with custom CSS
- **Backend**: PyTorch + Transformers
- **Optimization**: Unsloth for fast inference
- **Hardware**: NVIDIA L40S GPU (44.7GB VRAM)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support
- 8GB+ VRAM recommended (44GB for optimal performance)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd car-damage-assessment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run car_damage_app.py
```

4. **Access the app**
- Local: http://localhost:8501
- Network: http://[your-ip]:8501

## 📋 Requirements

### Core Dependencies
```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
unsloth>=2024.8.0
Pillow>=9.0.0
accelerate>=0.20.0
bitsandbytes>=0.41.0
datasets>=2.14.0
```

### Hardware Requirements
- **Minimum**: 8GB GPU memory
- **Recommended**: 24GB+ GPU memory
- **Optimal**: 40GB+ GPU memory (NVIDIA L40S)

## 🎮 Usage

### Basic Workflow
1. **Launch Application**: Run `streamlit run car_damage_app.py`
2. **Upload Image**: Use file uploader or camera capture
3. **Analyze Damage**: Click "Analyze Damage" button
4. **Review Results**: View detailed assessment and severity classification

### Supported Formats
- **Images**: PNG, JPG, JPEG
- **Max Size**: 200MB per image
- **Resolution**: Automatically resized to 1024x1024 for processing

### Analysis Output
The system provides:
- **Damage Location**: Specific car parts affected
- **Damage Type**: Scratches, dents, cracks, etc.
- **Severity Level**: Minor, Moderate, or Major
- **Professional Assessment**: Detailed description in French/English

## 🔧 Configuration

### GPU Memory Settings
The application automatically configures GPU memory limits:
- **Primary**: 40GB for L40S
- **Fallback**: 35GB for medium models
- **Minimal**: 30GB for base models

### Model Loading Strategy
1. Try Unsloth optimized model
2. Fallback to standard transformers
3. Load lighter model variants if memory insufficient
4. Use base model as final fallback

### Environment Variables
```bash
TORCH_COMPILE_DISABLE=1          # Disable compilation issues
TORCHDYNAMO_DISABLE=1           # Disable dynamo compilation
CUDA_LAUNCH_BLOCKING=1          # Synchronous CUDA for debugging
```

## 🏢 Project Structure

```
car-damage-assessment/
├── car_damage_app.py           # Main Streamlit application
├── interface.ipynb             # Gradio alternative interface
├── Learning copy.ipynb         # Clean inference notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── TROUBLESHOOTING.md          # Detailed problem solutions
├── run_app.bat                 # Windows batch runner
├── setup_and_run.py           # Setup automation script
├── car_damage_assessment_model/ # Trained model files
├── car_damage_model/           # Training checkpoints
└── models/                     # Model storage directory
```

## 🎨 UI Components

### Main Interface
- **Header**: Modern gradient title with professional styling
- **Sidebar**: System information and GPU monitoring
- **Upload Area**: Drag-and-drop file upload with hover effects
- **Analysis Panel**: Real-time processing with progress indicators
- **Results Display**: Styled output with severity color coding

### Styling Features
- Gradient backgrounds and modern color scheme
- Responsive design for different screen sizes
- Professional typography and spacing
- Interactive hover effects and animations
- Color-coded severity indicators (green/yellow/red)

## 🔬 Technical Details

### Model Architecture
- **Base**: Llama-3.2-11B-Vision-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit for memory efficiency
- **Inference**: Optimized with Unsloth framework

### Performance Optimizations
- **Memory Management**: Automatic GPU cache clearing
- **Image Processing**: Smart resizing and format conversion
- **Batch Processing**: Single image inference with optimal batching
- **Error Recovery**: Multiple fallback strategies

### Security & Reliability
- **Input Validation**: File type and size checking
- **Error Handling**: Comprehensive exception management
- **Memory Safety**: Automatic cleanup and monitoring
- **Fallback Systems**: Multiple model and inference strategies

## 📊 Performance Metrics

### Processing Speed
- **Average**: 10-30 seconds per image
- **GPU**: 5-15 seconds with optimal settings
- **CPU Fallback**: 30-60 seconds

### Accuracy
- **Professional Grade**: Trained on automotive damage datasets
- **Multi-category**: Comprehensive damage type detection
- **Severity Assessment**: Reliable classification system

## 🛠️ Development

### Local Development
1. Set up Python virtual environment
2. Install development dependencies
3. Configure GPU environment
4. Run application in development mode

### Debugging
- Enable CUDA_LAUNCH_BLOCKING for detailed error reports
- Monitor GPU memory usage in sidebar
- Check model loading status and fallbacks
- Review error logs in terminal output

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes with proper testing
4. Submit pull request with detailed description

## 🆘 Troubleshooting

For detailed troubleshooting of common issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Common quick fixes:
- **Memory Issues**: Reduce max_memory settings or use lighter models
- **CUDA Errors**: Enable CUDA_LAUNCH_BLOCKING and check GPU status
- **Model Loading**: Verify HuggingFace access and model availability
- **Performance**: Check GPU utilization and memory usage

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Unsloth**: For fast inference optimization
- **HuggingFace**: For model hosting and transformers library
- **Streamlit**: For the excellent web application framework
- **Meta**: For the Llama-3.2-Vision foundation model

## 📧 Contact

For questions, issues, or contributions, please open an issue on the repository or contact the development team.

---

**Built with ❤️ for professional automotive damage assessment**
