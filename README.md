# Misinformation Detection in Social Media posts using a BERT-based-uncased model with LIME interpretability


This project implements a BERT-based classification model to detect misinformation in news headlines, with LIME interpretability to explain model predictions.

## Project Description

The system fine-tunes a BERT model on the FakeNewsNet dataset to classify news headlines as real or fake. It includes comprehensive evaluation metrics and LIME (Local Interpretable Model-agnostic Explanations) visualizations to understand the model's decision-making process.

## Prerequisites

- Python 3.8+
- Google Colab (for GPU acceleration) or local environment with CUDA support

## Dependencies

Install the required packages:

```bash
pip install transformers torch scikit-learn matplotlib seaborn pandas lime tqdm
```

## Dataset

The code expects four CSV files from the FakeNewsNet dataset:
- gossipcop_fake.csv
- gossipcop_real.csv
- politifact_fake.csv
- politifact_real.csv

These files should be uploaded when prompted by the code.

## Project Structure

- `misinformation_detection_in_social_media_posts.py` - Main Python script containing the entire pipeline
- `misinformation_detection_in_social_media_posts.ipynb` - Jupyter notebook version for Google Colab
- `README.md` - This file
- `requirements.txt` - List of dependencies

## How to Run

### Using Google Colab (Recommended)

1. Upload the `misinformation_detection_in_social_media_posts.ipynb` to Google Colab
2. Ensure GPU acceleration is enabled (Runtime > Change runtime type > Hardware accelerator > GPU)
3. Run all cells in order
4. When prompted, upload the four CSV dataset files

### Using Local Environment

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the four CSV dataset files in the same directory as the script
3. Run the script:
   ```bash
   python misinformation_detection_in_social_media_posts.py
   ```

## Code Flow

1. Data loading and preprocessing
2. BERT model setup and fine-tuning
3. Model evaluation
4. LIME interpretability analysis

## Output

The code generates:
- Training progress metrics
- Final evaluation metrics (accuracy, F1, precision, recall, AUC-ROC)
- Confusion matrix visualization
- ROC curve
- Precision-Recall curve
- LIME visualizations showing feature importance for sample predictions
- Feature importance rankings

## Customization

You can modify these parameters in the code:
- `batch_size`: Change for different memory requirements
- `num_epochs`: Adjust training duration
- `lr`: Learning rate
- `num_features`: Number of features shown in LIME analysis

## Troubleshooting

- If you encounter GPU memory issues, reduce the batch size
- If LIME installation fails, run `!pip install lime` separately before importing
- For large datasets, consider using a subset for faster experimentation

## License

This project is provided for educational purposes only. Please respect the terms of use for the FakeNewsNet dataset.
