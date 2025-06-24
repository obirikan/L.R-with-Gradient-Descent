## 📈 Linear Regression with Gradient Descent

This project demonstrates how to use **gradient descent** to find the best-fitting line for a dataset using **manual calculations**. It visualizes how the algorithm learns to reduce error and draw the best line.

### 🔧 What the Code Does

* Manually calculates gradients (partial derivatives) for the slope (`m`) and intercept (`b`) of a line.
* Uses those gradients to update the values of `m` and `b` iteratively.
* Visualizes the final line that best fits the given data.

### 📊 Dataset

We use a simple dataset of months and revenue:

```python
months = [1, 2, 3, ..., 12]
revenue = [52, 74, 79, ..., 184]
```

### 🔄 How It Works

1. **Initialize**: Start with `m = 0` and `b = 0`.
2. **Loop**: For a number of iterations (1000), update `m` and `b` using the gradients of error:

   * `b_gradient = -2/N * sum(y - (mx + b))`
   * `m_gradient = -2/N * sum(x * (y - (mx + b)))`
3. **Update Rule**:

   ```python
   b = b - learning_rate * b_gradient
   m = m - learning_rate * m_gradient
   ```
4. **Prediction**: After training, use `y = mx + b` to generate predictions.
5. **Plot**: Use `matplotlib` to show the original points and the fitted line.

### 🖥 Output

* A graph showing:

  * Dots for the original data points
  * A line representing the learned model

### 📦 Requirements

```bash
pip install matplotlib
```

### ▶️ How to Run

```bash
python gradient_descent.py
```

### 📌 Key Concepts

* Gradient Descent
* Mean Squared Error
* Linear Regression
* Manual Optimization Without Libraries

### ✅ Example Result

Final model:

```python
y = 11.31x + 37.82
```

(Final values of `m` and `b` may vary based on learning rate and data)

---

> This is great for educational purposes and understanding how machine learning works under the hood before using tools like `scikit-learn` or `TensorFlow`.
