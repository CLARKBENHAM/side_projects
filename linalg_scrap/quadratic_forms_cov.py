# %% https://claude.ai/chat/590b6bdb-924f-4e27-a4db-fe9170f7d74f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

# Set random seed for reproducibility
np.random.seed(42)


# Function to plot a covariance ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    """
    if ax is None:
        ax = plt.gca()

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    # Width and height of ellipse
    width, height = 2 * nstd * np.sqrt(eigvals)

    # Draw the ellipse
    ellipse = patches.Ellipse(xy=pos, width=width, height=height, angle=np.degrees(theta), **kwargs)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

    # Return the basis vectors (scaled by singular values for visualization)
    basis_x = eigvecs[:, 0] * np.sqrt(eigvals[0])
    basis_y = eigvecs[:, 1] * np.sqrt(eigvals[1])

    return basis_x, basis_y


# Original covariance matrix and mean
mean = np.array([0, 0])
cov = np.array([[2.0, 0.8], [0.8, 1.0]])

# Generate multivariate Gaussian samples
n_samples = 1000
samples = np.random.multivariate_normal(mean, cov, size=n_samples)

# Create a random transformation matrix
A = np.array([[2.0, 1.0], [0.0, 1.5]])

# Apply the transformation to the samples
transformed_samples = samples @ A.T  # Using A.T to match with numpy's conventions

# Calculate the empirical covariance matrices
empirical_cov_orig = np.cov(samples, rowvar=False)
empirical_cov_trans = np.cov(transformed_samples, rowvar=False)

# Calculate the theoretical transformed covariance
theoretical_cov_trans = A @ cov @ A.T

# Set up the figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot original samples
axs[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
axs[0].set_title("Original Multivariate Gaussian with Basis Vectors")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].grid(True)
axs[0].axis("equal")

# Plot transformed samples
axs[1].scatter(transformed_samples[:, 0], transformed_samples[:, 1], alpha=0.5, s=5)
axs[1].set_title("Transformed Gaussian with Basis Vectors")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].grid(True)
axs[1].axis("equal")

# Plot covariance ellipses and get basis vectors
orig_basis_x, orig_basis_y = plot_cov_ellipse(
    empirical_cov_orig, mean, nstd=2, ax=axs[0], fc="none", ec="red", lw=2
)
trans_empirical_basis_x, trans_empirical_basis_y = plot_cov_ellipse(
    empirical_cov_trans,
    np.mean(transformed_samples, axis=0),
    nstd=2,
    ax=axs[1],
    fc="none",
    ec="red",
    lw=2,
)
trans_theoretical_basis_x, trans_theoretical_basis_y = plot_cov_ellipse(
    theoretical_cov_trans,
    np.mean(transformed_samples, axis=0),
    nstd=2,
    ax=axs[1],
    fc="none",
    ec="blue",
    ls="--",
    lw=2,
)

# Draw basis vectors (eigenvectors scaled by eigenvalues) for original distribution
center = mean
axs[0].arrow(
    center[0],
    center[1],
    orig_basis_x[0],
    orig_basis_x[1],
    head_width=0.1,
    head_length=0.2,
    fc="red",
    ec="red",
    label="1st Basis Vector",
)
axs[0].arrow(
    center[0],
    center[1],
    orig_basis_y[0],
    orig_basis_y[1],
    head_width=0.1,
    head_length=0.2,
    fc="green",
    ec="green",
    label="2nd Basis Vector",
)

# Draw empirical basis vectors for transformed distribution
center_trans = np.mean(transformed_samples, axis=0)
axs[1].arrow(
    center_trans[0],
    center_trans[1],
    trans_empirical_basis_x[0],
    trans_empirical_basis_x[1],
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="red",
    label="Empirical 1st Basis",
)
axs[1].arrow(
    center_trans[0],
    center_trans[1],
    trans_empirical_basis_y[0],
    trans_empirical_basis_y[1],
    head_width=0.2,
    head_length=0.3,
    fc="green",
    ec="green",
    label="Empirical 2nd Basis",
)

# Draw theoretical basis vectors for transformed distribution
axs[1].arrow(
    center_trans[0],
    center_trans[1],
    trans_theoretical_basis_x[0],
    trans_theoretical_basis_x[1],
    head_width=0.2,
    head_length=0.3,
    fc="darkred",
    ec="darkred",
    ls="--",
    label="Theoretical 1st Basis",
)
axs[1].arrow(
    center_trans[0],
    center_trans[1],
    trans_theoretical_basis_y[0],
    trans_theoretical_basis_y[1],
    head_width=0.2,
    head_length=0.3,
    fc="darkgreen",
    ec="darkgreen",
    ls="--",
    label="Theoretical 2nd Basis",
)

# Add legends
axs[0].legend()
axs[1].legend()

# Print the covariance matrices and comparison
print("Original Covariance Matrix:")
print(empirical_cov_orig)
print("\nTransformed Covariance Matrix (Empirical):")
print(empirical_cov_trans)
print("\nTransformed Covariance Matrix (Theoretical: A*Cov*A^T):")
print(theoretical_cov_trans)
print("\nDifference (Empirical - Theoretical):")
print(empirical_cov_trans - theoretical_cov_trans)

# Print the eigenvectors and eigenvalues
orig_eigvals, orig_eigvecs = np.linalg.eigh(empirical_cov_orig)
trans_eigvals, trans_eigvecs = np.linalg.eigh(empirical_cov_trans)
theo_eigvals, theo_eigvecs = np.linalg.eigh(theoretical_cov_trans)

print("\nOriginal Eigenvalues:", orig_eigvals)
print("Original Eigenvectors:\n", orig_eigvecs)

print("\nTransformed Eigenvalues (Empirical):", trans_eigvals)
print("Transformed Eigenvectors (Empirical):\n", trans_eigvecs)

print("\nTransformed Eigenvalues (Theoretical):", theo_eigvals)
print("Transformed Eigenvectors (Theoretical):\n", theo_eigvecs)

plt.tight_layout()
plt.show()

# Now, let's create a second figure that shows the effect of the transformation
# on the basis vectors directly
fig2, ax2 = plt.subplots(figsize=(8, 8))

# Draw original samples
ax2.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=5, color="blue", label="Original Samples")

# Draw transformed samples
ax2.scatter(
    transformed_samples[:, 0],
    transformed_samples[:, 1],
    alpha=0.1,
    s=5,
    color="red",
    label="Transformed Samples",
)

# Draw original basis vectors
ax2.arrow(
    0,
    0,
    orig_basis_x[0],
    orig_basis_x[1],
    head_width=0.1,
    head_length=0.2,
    fc="blue",
    ec="blue",
    label="Original 1st Basis",
)
ax2.arrow(
    0,
    0,
    orig_basis_y[0],
    orig_basis_y[1],
    head_width=0.1,
    head_length=0.2,
    fc="cyan",
    ec="cyan",
    label="Original 2nd Basis",
)

# Draw transformed basis vectors
ax2.arrow(
    0,
    0,
    trans_theoretical_basis_x[0],
    trans_theoretical_basis_x[1],
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="red",
    label="Transformed 1st Basis",
)
ax2.arrow(
    0,
    0,
    trans_theoretical_basis_y[0],
    trans_theoretical_basis_y[1],
    head_width=0.2,
    head_length=0.3,
    fc="pink",
    ec="pink",
    label="Transformed 2nd Basis",
)

# Add labels and grid
ax2.set_title("Original vs. Transformed Distributions and Basis Vectors")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True)
ax2.axis("equal")
ax2.legend()

plt.tight_layout()
plt.show()

# Let's calculate what happens when we directly transform the eigenvectors
# by multiplying them with A and compare with the eigenvectors of A*Cov*A^T

# Get the sorted eigenvectors of the original covariance
orig_eigvals, orig_eigvecs = np.linalg.eigh(cov)
idx = orig_eigvals.argsort()[::-1]
orig_eigvals = orig_eigvals[idx]
orig_eigvecs = orig_eigvecs[:, idx]

# Directly transform the original eigenvectors by multiplying with A,
direct_transformed_basis1 = A @ orig_eigvecs[:, 0]
direct_transformed_basis2 = A @ orig_eigvecs[:, 1]

# Get the eigenvectors of the theoretical transformed covariance matrix
theo_eigvals, theo_eigvecs = np.linalg.eigh(theoretical_cov_trans)
idx = theo_eigvals.argsort()[::-1]
theo_eigvals = theo_eigvals[idx]
theo_eigvecs = theo_eigvecs[:, idx]

# Now compare them
print("\nOriginal primary eigenvector:", orig_eigvecs[:, 0])
print("Direct transformation of primary eigenvector (wrong):", direct_transformed_basis1)
print("Primary eigenvector of transformed covariance:", theo_eigvecs[:, 0])
print("\nOriginal secondary eigenvector:", orig_eigvecs[:, 1])
print("Direct transformation of secondary eigenvector (wrong):", direct_transformed_basis2)
print("Secondary eigenvector of transformed covariance:", theo_eigvecs[:, 1])

# Create a third figure to visualize this comparison
fig3, ax3 = plt.subplots(figsize=(8, 8))

# Plot the directly transformed eigenvectors
ax3.arrow(
    0,
    0,
    direct_transformed_basis1[0],
    direct_transformed_basis1[1],
    head_width=0.2,
    head_length=0.3,
    fc="purple",
    ec="purple",
    label="Direct Transform of 1st Eigenvector",
)
ax3.arrow(
    0,
    0,
    direct_transformed_basis2[0],
    direct_transformed_basis2[1],
    head_width=0.2,
    head_length=0.3,
    fc="orange",
    ec="orange",
    label="Direct Transform of 2nd Eigenvector",
)

# Plot the eigenvectors of the transformed covariance
ax3.arrow(
    0,
    0,
    theo_eigvecs[0, 0] * np.sqrt(theo_eigvals[0]),
    theo_eigvecs[1, 0] * np.sqrt(theo_eigvals[0]),
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="red",
    ls="--",
    label="1st Eigenvector of Transformed Cov",
)
ax3.arrow(
    0,
    0,
    theo_eigvecs[0, 1] * np.sqrt(theo_eigvals[1]),
    theo_eigvecs[1, 1] * np.sqrt(theo_eigvals[1]),
    head_width=0.2,
    head_length=0.3,
    fc="green",
    ec="green",
    ls="--",
    label="2nd Eigenvector of Transformed Cov",
)

# Add some context with samples
ax3.scatter(transformed_samples[:, 0], transformed_samples[:, 1], alpha=0.1, s=5, color="gray")

# Add labels and grid
ax3.set_title("Comparison: Direct Transform vs. Eigenvectors of Transformed Covariance")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.grid(True)
ax3.axis("equal")
ax3.legend()

plt.tight_layout()
plt.show()

# %% Correct way to transform eigenvectors for transformed sample
# Get eigendecomposition of A
a_eigvals, a_eigvecs = np.linalg.eig(A)
print("Eigenvalues of A:", a_eigvals)
print("Eigenvectors of A:\n", a_eigvecs)

# Apply the transformation to the samples
transformed_samples = samples @ A.T  # Using A.T to match with numpy's conventions

# Calculate the empirical covariance matrices
empirical_cov_orig = np.cov(samples, rowvar=False)
empirical_cov_trans = np.cov(transformed_samples, rowvar=False)

# Calculate the theoretical transformed covariance
theoretical_cov_trans = A @ cov @ A.T

# Get the eigenvectors of the original covariance
orig_eigvals, orig_eigvecs = np.linalg.eigh(cov)
idx = orig_eigvals.argsort()[::-1]
orig_eigvals = orig_eigvals[idx]
orig_eigvecs = orig_eigvecs[:, idx]

# Get the eigenvectors of the transformed covariance matrix
trans_eigvals, trans_eigvecs = np.linalg.eigh(theoretical_cov_trans)
idx = trans_eigvals.argsort()[::-1]
trans_eigvals = trans_eigvals[idx]
trans_eigvecs = trans_eigvecs[:, idx]

# Set up the figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot original samples
axs[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
axs[0].set_title("Original Multivariate Gaussian with Basis Vectors")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].grid(True)
axs[0].axis("equal")

# Plot transformed samples
axs[1].scatter(transformed_samples[:, 0], transformed_samples[:, 1], alpha=0.5, s=5)
axs[1].set_title("Transformed Gaussian with Basis Vectors")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].grid(True)
axs[1].axis("equal")

# Plot covariance ellipses and get basis vectors
orig_basis_x, orig_basis_y = plot_cov_ellipse(
    cov, mean, nstd=2, ax=axs[0], fc="none", ec="red", lw=2
)
trans_theoretical_basis_x, trans_theoretical_basis_y = plot_cov_ellipse(
    theoretical_cov_trans, np.zeros(2), nstd=2, ax=axs[1], fc="none", ec="blue", ls="--", lw=2
)

# Draw basis vectors (eigenvectors scaled by eigenvalues) for original distribution
axs[0].arrow(
    0,
    0,
    orig_basis_x[0],
    orig_basis_x[1],
    head_width=0.1,
    head_length=0.2,
    fc="red",
    ec="red",
    label="1st Basis Vector",
)
axs[0].arrow(
    0,
    0,
    orig_basis_y[0],
    orig_basis_y[1],
    head_width=0.1,
    head_length=0.2,
    fc="green",
    ec="green",
    label="2nd Basis Vector",
)

# Draw theoretical basis vectors for transformed distribution
axs[1].arrow(
    0,
    0,
    trans_theoretical_basis_x[0],
    trans_theoretical_basis_x[1],
    head_width=0.2,
    head_length=0.3,
    fc="darkred",
    ec="darkred",
    label="Transformed 1st Basis",
)
axs[1].arrow(
    0,
    0,
    trans_theoretical_basis_y[0],
    trans_theoretical_basis_y[1],
    head_width=0.2,
    head_length=0.3,
    fc="darkgreen",
    ec="darkgreen",
    label="Transformed 2nd Basis",
)

# Add legends
axs[0].legend()
axs[1].legend()

plt.tight_layout()
plt.show()

# Now let's compare different approaches to transforming the basis vectors

# Create a figure to visualize the comparison
fig2, ax2 = plt.subplots(figsize=(10, 10))

# Plot the transformed samples for reference
ax2.scatter(transformed_samples[:, 0], transformed_samples[:, 1], alpha=0.1, s=5, color="gray")

# 1. Directly transformed eigenvectors (incorrect approach)
direct_transformed_basis1 = A @ orig_eigvecs[:, 0]
direct_transformed_basis2 = A @ orig_eigvecs[:, 1]

# 2. Correct transformation: eigenvectors of transformed covariance
# These are already computed as trans_eigvecs

# 3. Using eigenvectors of A to explain the transformation
# Project the original eigenvectors onto the eigenvectors of A
# and then transform using the eigenvalues of A

# Draw original basis vectors
ax2.arrow(
    0,
    0,
    orig_basis_x[0],
    orig_basis_x[1],
    head_width=0.2,
    head_length=0.3,
    fc="black",
    ec="black",
    label="Original Basis Vector 1",
)
ax2.arrow(
    0,
    0,
    orig_basis_y[0],
    orig_basis_y[1],
    head_width=0.2,
    head_length=0.3,
    fc="gray",
    ec="gray",
    label="Original Basis Vector 2",
)

# Draw the directly transformed vectors
ax2.arrow(
    0,
    0,
    direct_transformed_basis1[0],
    direct_transformed_basis1[1],
    head_width=0.2,
    head_length=0.3,
    fc="blue",
    ec="blue",
    label="Direct Transform (A·v1)",
)
ax2.arrow(
    0,
    0,
    direct_transformed_basis2[0],
    direct_transformed_basis2[1],
    head_width=0.2,
    head_length=0.3,
    fc="cyan",
    ec="cyan",
    label="Direct Transform (A·v2)",
)

# Draw the eigenvectors of the transformed covariance (correct approach)
ax2.arrow(
    0,
    0,
    trans_eigvecs[0, 0] * np.sqrt(trans_eigvals[0]),
    trans_eigvecs[1, 0] * np.sqrt(trans_eigvals[0]),
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="red",
    label="Eigenvector of A·Σ·A^T (v1)",
)
ax2.arrow(
    0,
    0,
    trans_eigvecs[0, 1] * np.sqrt(trans_eigvals[1]),
    trans_eigvecs[1, 1] * np.sqrt(trans_eigvals[1]),
    head_width=0.2,
    head_length=0.3,
    fc="green",
    ec="green",
    label="Eigenvector of A·Σ·A^T (v2)",
)

# Draw the eigenvectors of A (to show their role)
for i, (val, vec) in enumerate(zip(a_eigvals, a_eigvecs.T)):
    scaled_vec = vec * 3  # Scale for visibility
    ax2.arrow(
        0,
        0,
        scaled_vec[0],
        scaled_vec[1],
        head_width=0.2,
        head_length=0.3,
        fc="purple",
        ec="purple",
        ls="--",
        label=f"Eigenvector {i+1} of A (λ={val:.2f})",
    )

# Add labels and grid
ax2.set_title("Comparison of Different Transformation Approaches")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True)
ax2.axis("equal")
ax2.legend()

# Print numerical comparisons
print("\nOriginal Covariance Eigenvectors:")
print(orig_eigvecs)
print("\nTransformed Covariance Eigenvectors (from A·Σ·A^T):")
print(trans_eigvecs)
print("\nDirect Transformation of Original Eigenvectors (incorrect approach):")
print("A·v1 =", direct_transformed_basis1)
print("A·v2 =", direct_transformed_basis2)

# Compute and print the special case where A is diagonal with eigenvectors aligned with standard basis
print(
    "\nSpecial Case - If A is diagonal or its eigenvectors align with the covariance eigenvectors:"
)
# In this case, the transformation would preserve the eigenvector directions
# and just scale them by the corresponding eigenvalues of A

plt.tight_layout()
plt.show()

# Let's demonstrate a special case where A's eigenvectors align with the covariance eigenvectors
print("\n--- SPECIAL CASE DEMONSTRATION ---")
print(
    "Creating a special transformation matrix with eigenvectors aligned with covariance"
    " eigenvectors"
)

# Create a transformation matrix whose eigenvectors align with the covariance eigenvectors
# This is done by A = Q·D·Q^T where Q contains the eigenvectors of the covariance
# and D is a diagonal matrix of desired eigenvalues
special_eigenvalues = np.array([3.0, 1.5])  # Arbitrary eigenvalues for demonstration
special_A = orig_eigvecs @ np.diag(special_eigenvalues) @ orig_eigvecs.T

print("Special A matrix:")
print(special_A)
print("Eigenvalues of special A:", np.linalg.eigvals(special_A))
print("Eigenvectors of special A:")
print(np.linalg.eig(special_A)[1])

# Apply this special transformation
special_transformed_samples = samples @ special_A.T
special_theoretical_cov = special_A @ cov @ special_A.T

# Get eigenvectors of the special transformed covariance
special_trans_eigvals, special_trans_eigvecs = np.linalg.eigh(special_theoretical_cov)
idx = special_trans_eigvals.argsort()[::-1]
special_trans_eigvals = special_trans_eigvals[idx]
special_trans_eigvecs = special_trans_eigvecs[:, idx]

# Direct transformation of original eigenvectors with special A
special_direct_transformed_basis1 = special_A @ orig_eigvecs[:, 0]
special_direct_transformed_basis2 = special_A @ orig_eigvecs[:, 1]

print("\nOriginal eigenvectors:")
print(orig_eigvecs)
print("\nEigenvectors of special transformed covariance:")
print(special_trans_eigvecs)
print("\nDirectly transformed eigenvectors with special A:")
print("A·v1 =", special_direct_transformed_basis1)
print("A·v2 =", special_direct_transformed_basis2)

# Create a figure to visualize the special case
fig3, ax3 = plt.subplots(figsize=(10, 10))

# Plot the special transformed samples
ax3.scatter(
    special_transformed_samples[:, 0],
    special_transformed_samples[:, 1],
    alpha=0.1,
    s=5,
    color="gray",
)

# Plot original basis vectors
ax3.arrow(
    0,
    0,
    orig_basis_x[0],
    orig_basis_x[1],
    head_width=0.2,
    head_length=0.3,
    fc="black",
    ec="black",
    label="Original Basis Vector 1",
)
ax3.arrow(
    0,
    0,
    orig_basis_y[0],
    orig_basis_y[1],
    head_width=0.2,
    head_length=0.3,
    fc="gray",
    ec="gray",
    label="Original Basis Vector 2",
)

# Plot directly transformed vectors (should align with eigenvectors in this special case)
ax3.arrow(
    0,
    0,
    special_direct_transformed_basis1[0],
    special_direct_transformed_basis1[1],
    head_width=0.2,
    head_length=0.3,
    fc="blue",
    ec="blue",
    label="Direct Transform (Special A·v1)",
)
ax3.arrow(
    0,
    0,
    special_direct_transformed_basis2[0],
    special_direct_transformed_basis2[1],
    head_width=0.2,
    head_length=0.3,
    fc="cyan",
    ec="cyan",
    label="Direct Transform (Special A·v2)",
)

# Plot eigenvectors of special transformed covariance
ax3.arrow(
    0,
    0,
    special_trans_eigvecs[0, 0] * np.sqrt(special_trans_eigvals[0]),
    special_trans_eigvecs[1, 0] * np.sqrt(special_trans_eigvals[0]),
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="red",
    ls="--",
    label="Eigenvector of Special A·Σ·A^T (v1)",
)
ax3.arrow(
    0,
    0,
    special_trans_eigvecs[0, 1] * np.sqrt(special_trans_eigvals[1]),
    special_trans_eigvecs[1, 1] * np.sqrt(special_trans_eigvals[1]),
    head_width=0.2,
    head_length=0.3,
    fc="green",
    ec="green",
    ls="--",
    label="Eigenvector of Special A·Σ·A^T (v2)",
)

# Add labels and grid
ax3.set_title("Special Case: A with Eigenvectors Aligned with Covariance Eigenvectors")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.grid(True)
ax3.axis("equal")
ax3.legend()

plt.tight_layout()
plt.show()
