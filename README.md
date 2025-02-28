# Mixture-Density-Network-for-Credit-Risk-Assessment


# CreditScoreMDNModel

This repository contains a PyTorch implementation of a Mixture Density Network (MDN) tailored for credit score prediction, leveraging both categorical and continuous features.

## Overview

The `CreditScoreMDNModel` class is a neural network designed to model the conditional probability distribution of a credit score given various input features. It utilizes an MDN to capture potential multi-modal distributions, which are common in credit scoring due to varying customer profiles and risk factors.

## Features

* **Categorical Feature Embeddings:** Handles categorical inputs through embedding layers, allowing the model to learn meaningful representations of discrete features.
* **Continuous Feature Integration:** Seamlessly combines continuous features with embedded categorical features.
* **Mixture Density Network (MDN):** Outputs a mixture of Gaussian distributions, enabling the model to represent complex, multi-modal credit score distributions.
* **Customizable Architecture:** Parameters such as embedding dimensions, hidden layer sizes, and the number of mixture components can be easily configured.
* **Dropout Regularization:** Includes dropout layers to prevent overfitting.
* **Softplus for Standard Deviation:** Ensures the standard deviation of the Gaussian components is always positive and numerically stable.

## Model Architecture

The model consists of the following components:

1.  **Embedding Layers:**
    * `nn.Embedding` layers for each categorical input feature.
2.  **Feature Concatenation:**
    * Concatenates the embedded categorical features and continuous features.
3.  **Base Network:**
    * A series of linear layers with SiLU activation and dropout for feature extraction.
4.  **MDN Output Layers:**
    * `pi_layer`: Linear layer to produce mixing coefficients (probabilities) for each Gaussian component.
    * `mu_layer`: Linear layer to produce the means of each Gaussian component.
    * `sigma_layer`: Linear layer to produce the standard deviations of each Gaussian component (using `F.softplus` for positivity).

## Usage

### Installation

To use this model, ensure you have PyTorch installed.




### Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CreditScoreMDNModel(nn.Module):
    def __init__(self, 
                 num_categories1, embedding_dim1, 
                 num_categories2, embedding_dim2, 
                 continuous_dim,  
                 hidden_dim, 
                 num_components=2):
        super(CreditScoreMDNModel, self).__init__()
        
        # Embedding layers for each categorical variable
        self.embedding1 = nn.Embedding(num_categories1, embedding_dim1)
        self.embedding2 = nn.Embedding(num_categories2, embedding_dim2)
        
        # Compute the combined input dimension
        total_embedding_dim = embedding_dim1 + embedding_dim2
        input_dim = total_embedding_dim + continuous_dim
        
        # Base network to extract features
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        # MDN output layers
        self.pi_layer = nn.Linear(hidden_dim // 2, num_components)  # mixing coefficients
        self.mu_layer = nn.Linear(hidden_dim // 2, num_components)   # expected value of the normal
        self.sigma_layer = nn.Linear(hidden_dim // 2, num_components)  # standard deviation of the normal
        
    def forward(self, categorical_input1, categorical_input2, continuous_input):
        # Process categorical inputs
        embedded1 = self.embedding1(categorical_input1)  # (batch_size, embedding_dim1)
        embedded2 = self.embedding2(categorical_input2)  # (batch_size, embedding_dim2)
        x_cat = torch.cat([embedded1, embedded2], dim=1)
        
        # Concatenate with continuous features
        x = torch.cat([x_cat, continuous_input], dim=1)
        
        # Extract shared features
        h = self.base(x)
        
        # Mixing coefficients
        pi = self.pi_layer(h)              
        pi = F.softmax(pi, dim=1)  # Ensure they sum to 1
        
        # Means for each component
        mu = self.mu_layer(h)
        
        # Standard deviations (using softplus to enforce positivity and numerical stability)
        sigma = self.sigma_layer(h)
        sigma = F.softplus(sigma) + 1e-6  # small constant to avoid exact zero
        
        return pi, mu, sigma

# Example Usage
num_categories1 = 10
embedding_dim1 = 5
num_categories2 = 5
embedding_dim2 = 3
continuous_dim = 2
hidden_dim = 32
num_components = 3

model = CreditScoreMDNModel(num_categories1, embedding_dim1, num_categories2, embedding_dim2, continuous_dim, hidden_dim, num_components)

# Create dummy input data
batch_size = 4
categorical_input1 = torch.randint(0, num_categories1, (batch_size,))
categorical_input2 = torch.randint(0, num_categories2, (batch_size,))
continuous_input = torch.randn(batch_size, continuous_dim)

pi, mu, sigma = model(categorical_input1, categorical_input2, continuous_input)

print("Mixing coefficients (pi):", pi)
print("Means (mu):", mu)
print("Standard deviations (sigma):", sigma)
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug fixes, feature requests, or improvements.

## License

This project is licensed under the MIT License.
```
