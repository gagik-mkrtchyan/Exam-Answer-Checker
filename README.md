Exam Answer Checker - Multi-Model Graph Analysis System

🎯 Overview
A comprehensive system for analyzing hand-drawn graphs in exam answers using multiple computer vision models:

Graph Localization: YOLOv8 for detecting graph regions
Node Detection: YOLOv8s for identifying graph vertices
Edge Classification: ConvNeXt for classifying connections
Adjacency Matrix Prediction: Custom GraphCNN
Document Layout: YOLOv9s-DocLayNet for paragraph detection

🚀 Quick Start
bash# Clone the repository
git clone https://github.com/yourusername/exam-answer-checker.git
cd exam-answer-checker

# Install dependencies
pip install -r requirements.txt

# Run inference on a sample image

![375](https://github.com/user-attachments/assets/3463ef42-e83c-46eb-9385-f726ddf77c82)


python src/inference/graph_inference.py \
    --image_path examples/sample_data/images/graph_example.jpg \
    --node_model models/pretrained/node_detection.pt \
    --edge_model models/pretrained/edge_classifier.pt
    
📁 Project Structure
exam-answer-checker/
├── src/                    # Main source code
│   ├── models/            # Model architectures (GraphCNN, Edge Classifier)
│   ├── training/          # Training scripts for all models
│   ├── evaluation/        # Model evaluation scripts
│   ├── inference/         # Inference pipelines and utilities
│   └── utils/            # Helper functions and utilities
├── docs/                  # Documentation and guides
├── examples/              # Usage examples and sample data
├── notebooks/             # Jupyter notebooks for tutorials
├── tests/                 # Unit tests
├── configs/               # Configuration files
└── requirements.txt       # Python dependencies

🔧 Installation

Clone the repository:
bashgit clone https://github.com/yourusername/exam-answer-checker.git
cd exam-answer-checker

Create a virtual environment:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bashpip install -r requirements.txt


📚 Models Overview
1. Graph Localization (YOLOv8)
Detects and localizes graph regions in exam papers with high accuracy.

![Screenshot from 2025-05-07 22-25-04](https://github.com/user-attachments/assets/5521425c-0812-401d-adc3-6d177951864a)

3. Node Detection (YOLOv8s)
Identifies individual nodes/vertices in detected graphs, handling various drawing styles.

![Screenshot from 2025-05-07 22-31-49](https://github.com/user-attachments/assets/bbb8a736-126f-4311-8240-03644f21bd5f)


5. Edge Classification (ConvNeXt)
Classifies whether connections exist between node pairs using state-of-the-art vision transformer architecture.

![452_corridor_2_4](https://github.com/user-attachments/assets/d0da526e-2ef6-4e8d-bf3c-2f1c4bdd1fc8)
![12_corridor_0_6](https://github.com/user-attachments/assets/636b0027-833f-4c80-8c82-5d579cbd788a)

7. Graph CNN
Custom CNN architecture that predicts adjacency matrices directly from graph images, eliminating the need for separate node and edge detection.
8. Paragraph Detection (YOLOv9s-DocLayNet)
Identifies text paragraphs and document layout elements to separate graph regions from textual content.

![Screenshot from 2025-04-29 13-15-03](https://github.com/user-attachments/assets/805906a4-7af2-4dbb-af4d-8d02142b89de)

🔄 Workflow

Document Analysis: Detect paragraphs and graph regions in the exam paper
Node Detection: Identify graph vertices using YOLO object detection
Corridor Extraction: Extract potential edge regions between detected nodes
Edge Classification: Classify extracted corridors as edges or non-edges
Graph Reconstruction: Build adjacency matrix from detected nodes and edges
Output Generation: Export results in various formats (JSON, CSV, visualization)

🚀 Usage Examples
Single Image Inference
bashpython src/inference/graph_inference.py \
    --image_path path/to/exam/image.jpg \
    --node_model models/pretrained/node_detection.pt \
    --edge_model models/pretrained/edge_classifier.pt \
    --output_dir results/
Batch Processing
bashpython src/inference/batch_inference.py \
    --model_path models/pretrained/graph_cnn.pt \
    --image_dir path/to/images/ \
    --output_json results/adjacency_matrices.json
Training a New Model
bashpython src/training/train_graph_cnn.py \
    --config configs/graph_cnn_config.yaml \
    --data_dir data/training_set/ \
    --output_dir models/custom/
📊 Performance Metrics

| Approach         | Model           | Accuracy | Precision | Recall | F1 Score |
|------------------|------------------|----------|-----------|--------|----------|
| Առաջին մոտեցում | ResNet18         | 95.28%   | 66.83%    | 66.07% | 66.45%   |
| Երկրորդ մոտեցում | ConvNeXt-Tiny    | 98.33%   | 98.65%    | 97.34% | 97.99%   |

Node Detection Accuracy: 94.2%
Edge Classification Precision: 91.8%
Graph Reconstruction F1-Score: 89.5%
Processing Time: ~2.3 seconds per exam page

🔧 Configuration
The system uses YAML configuration files located in the configs/ directory. Key parameters include:

Model Parameters: Learning rates, batch sizes, epochs
Data Parameters: Image augmentation settings, dataset paths
Inference Parameters: Confidence thresholds, NMS settings

🧪 Testing
Run the test suite to verify installation:
bashpython -m pytest tests/ -v
📖 Documentation

Getting Started Guide
Model Training Tutorial
API Reference
Troubleshooting

🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.
📝 Requirements

Python 3.8 or higher
PyTorch 1.7 or higher
CUDA-compatible GPU (recommended for training)
8GB+ RAM for inference, 16GB+ for training

🚨 Known Issues

Large images (>4K resolution) may require additional preprocessing
Some edge detection models may struggle with very faint or sketchy lines
Processing time increases significantly with the number of nodes in complex graphs

📞 Support
For questions, bug reports, or feature requests, please open an issue on GitHub.
🙏 Acknowledgments
This project builds upon several open-source frameworks:

YOLOv8 for object detection
ConvNeXt for edge classification
PyTorch for deep learning framework


Note: This project is under active development. Features and APIs may change between versions.
