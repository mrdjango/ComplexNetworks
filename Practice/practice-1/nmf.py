"""
 در کدی که ارائه داده ایم
 از خوشه‌بندی فازی به وسیله الگوریتم
 NMF (Non-Negative Matrix Factorization)
 برای دیتاست 
 Zachary
 استفاده شده است.
 //////////////
 NMF (Non-Negative Matrix Factorization)
 یک الگوریتم یادگیری ماتریسی است که به خصوص برای تجزیه ماتریس‌ها به ماتریس‌های مثبت نامنفی مفید است.
 /////////////////
 در اینجا، ما از 
 NMF 
 برای تجزیه داده‌های 
 Zachary
 استفاده می‌کنیم.
"""

import numpy as np
from sklearn.decomposition import NMF
from datasets import zachary_dataset
import matplotlib.pyplot as plt

# Load the Zachary dataset
X = np.array(zachary_dataset())

# Fit the NMF model
model = NMF(n_components=2, init="random", random_state=0)
W = model.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 6))

# Scatter plot for the first cluster (component 0)
plt.scatter(
    W[:, 0],
    W[:, 1],
    c=model.transform(X)[:, 0],
    cmap="viridis",
    label="Cluster 0",
    s=100,
    alpha=0.5,
)

# Scatter plot for the second cluster (component 1)
plt.scatter(
    W[:, 0],
    W[:, 1],
    c=model.transform(X)[:, 1],
    cmap="plasma",
    label="Cluster 1",
    s=100,
    alpha=0.5,
)

# Add labels and legend
plt.xlabel("Component 0")
plt.ylabel("Component 1")
plt.title("NMF Clustering of Zachary Dataset")
plt.legend()

# Show the plot
plt.show()


"""
این الگوریتم برای داده‌های 
Zachary
با تعداد خوشه‌های دوتایی 
(n_components=2)
اجرا شده است.
ما میتوانیم تعداد خوشه‌ها و سایر پارامترها را تغییر دهیم تا با توجه به مسئله‌ی خود خوشه‌بندی دلخواهی انجام دهیم.
"""
