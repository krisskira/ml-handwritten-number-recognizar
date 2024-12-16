# Calculator from ML Image-Reconizer

This is a simple app to calculate numbers with images.

<img style="margin:auto; width:600px;" src="./docs/images/ml-caculator-with-handwrite-reconizer.png" alt="Screenshot" width="600">


## Model training

To train the model, run the `model-training.py` script:

```bash
python model-training.py
```

The model will be saved in the `number_reconizer.keras` file.
for more information about the model, check the [model-training.ipynb notebook.](./docs/training.ipynb)

## Requirements

- Python 3.8+
- Virtualenv
- Pip

## Installation

1. Clone the repository:

```bash
git clone https://github.com/krisskira/ml-handwritten-number-recognizar.git
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

```bash
source venv/bin/activate
```

4. Install the requirements:

```bash
pip install -r requirements.txt
```

5. Run the app:

```bash
python server.py
```

## Usage Quickstart

Open your browser and go to `http://localhost:8000/`.

