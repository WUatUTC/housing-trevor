# Hamilton County Housing Value Predictor

This project uses a Deep Neural Network to estimate housing values
based on several property characteristics.

Inputs include:
- Land area (acres)
- Land value
- Building value
- Yard items value

The trained TensorFlow model is deployed using Streamlit to provide
an interactive web interface for predictions.

This application is intended for educational purposes only.
Predictions are approximate and depend on the training data used
to build the model.

# Model Behavior

The neural network provides approximate predictions based on patterns
learned in the training dataset. Because the model was trained primarily
on homes within a certain value range, predictions may not scale
proportionally when inputs exceed that range.

This limitation is common in machine learning models that extrapolate
beyond their training data. Machine learning models typically perform
best when making predictions within the range of data they were trained on.