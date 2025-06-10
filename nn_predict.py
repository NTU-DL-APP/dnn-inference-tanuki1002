import numpy as np
import json

# === Activation functions ===
def relu(x):
    # Implement the Rectified Linear Unit
    return np.maximum(0, x)

def softmax(x):
    # Implement the SoftMax function
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for i, layer in enumerate(model_arch["layers"]):
        ltype = layer["type"]
        cfg = layer["config"]

        if ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            # 權重鍵名稱遵照保存的格式，例如：layer_1_weights、layer_1_bias
            W = weights[f"layer_{i}_weights"]
            b = weights[f"layer_{i}_bias"]
            x = dense(x, W, b)

            activation = cfg.get("activation")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)

# === Testing ===
# 載入架構和權重
with open("/kaggle/working/model/fashion_mnist.json") as f:
    model_arch = json.load(f)
weights = np.load("/kaggle/working/model/fashion_mnist.npz")

# Load a dummy image (batch size 1)
# Make sure it's shape: (1, 28, 28, 1)
dummy_input = np.random.rand(1, 28*28).astype(np.float32)
output = nn_inference(model_arch, weights, dummy_input)

print("🧠 Output probabilities:", output)
print("✅ Predicted class:", np.argmax(output, axis=-1))