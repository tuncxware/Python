import numpy as np
import matplotlib.pyplot as plt

# 1. Training data
# Study hours
X = np.array([1, 2, 3, 4, 5, 6], dtype=float)

# Results: 0 = Fail, 1 = Pass
y = np.array([0, 0, 1, 0, 1, 1], dtype=float)

# 2. Initialize coefficients
b0 = 0.0
b1 = 0.0

# 3. Set training parameters
learning_rate = 0.1
epochs = 10000
n = len(X)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot the fitted probability curve as it changes during training.
plot_x = np.linspace(0, max(X), 200)
plt.ion()
fig, ax = plt.subplots()
ax.scatter(X, y, color="blue", label="Training data")
fit_line, = ax.plot(
    plot_x,
    sigmoid(b0 + b1 * plot_x),
    color="red",
    label="Fitted probability",
)
ax.set(
    xlabel="Study hours",
    ylabel="Probability of passing",
    title="Logistic Regression - Epoch 0",
    ylim=(-0.1, 1.1),
)
ax.legend()
plt.show(block=False)

# Training loop
for epoch in range(epochs):

    # Step 1: Calculate linear scores
    z = b0 + b1 * X

    # Step 2: Calculate probabilities
    p = sigmoid(z)

    # Step 3: Calculate log loss
    epsilon = 1e-15
    p_safe = np.clip(p, epsilon, 1 - epsilon)

    loss = -np.mean(
        y * np.log(p_safe)
        + (1 - y) * np.log(1 - p_safe)
    )

    # Step 4: Calculate gradients
    db0 = np.mean(p - y)
    db1 = np.mean((p - y) * X)

    # Step 5: Update coefficients
    b0 = b0 - learning_rate * db0
    b1 = b1 - learning_rate * db1

    # Update the fitted curve and report progress every 2000 iterations.
    if (epoch + 1) % 2000 == 0:
        fit_line.set_ydata(sigmoid(b0 + b1 * plot_x))
        ax.set_title(f"Logistic Regression - Epoch {epoch + 1}")
        fig.canvas.draw_idle()
        plt.pause(2.5)
        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {loss:.4f}, "
            f"b0: {b0:.4f}, "
            f"b1: {b1:.4f}"
        )

plt.ioff()
plt.show()

# 4. Make a prediction
study_hours = 4.5

z_new = b0 + b1 * study_hours
probability = sigmoid(z_new)

prediction = 1 if probability >= 0.5 else 0

print("\nFinal coefficients:")
print("Intercept (b0):", round(b0, 4))
print("Coefficient (b1):", round(b1, 4))

print("\nPrediction for 4.5 study hours:")
print("Probability of passing:", round(probability, 4))
print("Predicted class:", prediction)