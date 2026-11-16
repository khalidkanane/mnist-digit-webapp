# Handwritten Digit Recognizer Web App

Draw a digit (0-9) on a canvas in your browser and a CNN trained on MNIST predicts it. Built with PyTorch + Streamlit.

## Quick Start

```bash
git clone https://github.com/khalidkanane/mnist-digit-webapp.git
cd mnist-digit-webapp
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py