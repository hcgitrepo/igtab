# igtab: Integrated Gradients for Tabular Data

This is an implementation of Integrated Gradients for tabular data.

Simply import the **igtab** module in your Python code, and generate results from your DL model.

This module:
- Supports 2 initializing methods: zeros, random values from the uniform distribution
- Supports 4 types of models: binary classification, multiclass classification, regression, autoencoder for anomaly detection

Here are useful documents.
- Implementation: https://github.com/hcgitrepo/igtab/blob/main/Implementation_of_Integrated_Gradients_for_tabular_data.ipynb
- Example: https://github.com/hcgitrepo/igtab/blob/main/Example_of_Integrated_Gradients_for_tabular_data.ipynb
- Source: https://github.com/hcgitrepo/igtab/blob/main/igtab.py

## Usage

### Example of binary classification

```python
from igtab import Explainer
# "model" is your tensorflow model
# remove the last layer, sigmoid activation.
model2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
ig_deep = TabularExplainer(model=model2, type='sigmoid', base='zero')
res = ig_deep.ig_values(x_train_std)
```
