## 📈 Linear Regression with Gradient Descent

This project demonstrates how to use **gradient descent** to find the best-fitting line for a dataset using **manual calculations** (without any machine learning libraries like `scikit-learn`). It also visualizes how the algorithm learns over time by reducing the prediction error and drawing a better line.

---

### 📊 Dataset

We use a simple dataset of `months` and corresponding `revenue`:

```python
months = [1, 2, 3, ..., 12]
revenue = [52, 74, 79, ..., 184]
```

The goal is to find a line of the form:

```python
y = m * x + b
```

that best predicts `revenue` from `months`.

---

### 🔄 How Gradient Descent Works

Gradient Descent is an algorithm that tries to minimize the prediction error by adjusting the slope (`m`) and intercept (`b`) repeatedly:

1. **Initialize**: Start with `m = 0` and `b = 0` (a flat line).
2. **Predict**: Compute predicted values using the current line: `y_pred = m*x + b`
3. **Error**: Calculate the difference between actual and predicted values: `error = y - y_pred`
4. **Compute Gradients**:

   * For `b`: how the error changes with respect to the intercept
   * For `m`: how the error changes with respect to the slope
5. **Update `m` and `b`** using the gradients and a `learning_rate`
6. **Repeat** this process for many iterations (epochs)

---

### 💡 Intuition Behind Each Update

At the start, both `m` and `b` are `0`. That means the model predicts `y = 0` for every `x`. This leads to large errors, so the gradients are large. As we update `m` and `b` using these gradients, the predictions improve and the gradients shrink — meaning smaller steps over time.

This process leads us toward a line that minimizes the **Mean Squared Error (MSE)**.

---

### 🔢 Code Workflow

```python
# Gradient calculation functions
get_gradient_at_b()   # Calculates gradient with respect to b
get_gradient_at_m()   # Calculates gradient with respect to m

# Step function to update b and m
step_gradient()       # Performs one gradient descent step

gradient_descent()    # Loops over multiple steps to learn the final b and m
```

Each data point in the dataset contributes to the total error and therefore influences how much we update `m` and `b` at every iteration.

---

### 📈 Visual Output

After training, we use:

```python
y = m * x + b
```

to predict the outputs and plot them alongside the real data.

The `matplotlib` plot displays:

* Blue dots: original data
* Red line: final best-fit line learned via gradient descent

---

### 📌 Example Final Result

After 1000 iterations:

```python
y = 11.31 * x + 37.82
```

These are the final values of `m` and `b`, giving us the best-fit line.

---

### 📦 Requirements

```bash
pip install matplotlib
```

---

### ▶️ How to Run

```bash
python gradient_descent.py
```

---

### ✅ Concepts You’ll Learn

* Linear Regression
* Gradient Descent
* Mean Squared Error (MSE)
* How models learn from data step-by-step
* How slopes and intercepts update during training

---

> This project is ideal for learning **how machine learning really works under the hood**, before using higher-level libraries like `scikit-learn`, `TensorFlow`, or `PyTorch`.
