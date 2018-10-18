import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from sklearn.cluster import KMeans

NUM_CLUSTER = 20

sunset = imageio.imread('smallsunset.jpg')
strelitzia = imageio.imread('smallstrelitzia.jpg')
robert = imageio.imread('RobertMixed03.jpg')
chosen_image = sunset

def multi_normal(dist):
    return np.exp(-1/2*dist)/np.sqrt((2*np.pi)**3)

def logsumexp(X):
    x_max = X.max(1)
    return x_max + np.log(np.exp(X - x_max[:, None]).sum(1))

def alpha(X, w):
    pi = np.sum(w, axis=0) / w.shape[0] 
    cluster_mean = X @ w / np.sum(w, axis=0)
    
    all_mat = np.zeros((NUM_CLUSTER, num_pixel))
    for i in range(all_mat.shape[0]):
        all_mat[i] = -0.5 * np.sum((X.T - cluster_mean[:,i])**2, axis=1) + np.log(pi[i])
    
    return all_mat, cluster_mean

def compute_Wi(X, Mu, Pi, sample_idx):
    x_i = X[sample_idx]
    
    all_dist = np.linalg.norm(x_i - Mu, axis=1)
    all_dist -= np.min(all_dist)
    px = multi_normal(all_dist) 
    Wi = px / np.sum(px)
    return Wi


X = chosen_image.reshape(-1, 3).T
num_pixel = chosen_image.shape[0] * chosen_image.shape[1]
original_shape = chosen_image.shape

kmeans = KMeans(n_clusters=NUM_CLUSTER, random_state=0).fit(X.T)
labels = kmeans.labels_
k_mean_w = np.zeros((num_pixel, NUM_CLUSTER))
for i in range(NUM_CLUSTER):
    k_mean_w[np.where(labels==i),i] = 1

mu = None
converged = False
new_q_value = None
iteration = 0
w = k_mean_w.copy()
while True:
    A, mu = alpha(X, w)
    A = A.T
    q_value = np.sum(A * w) if new_q_value is None else new_q_value
    print("{}th Q-value: {}".format(iteration, q_value)) 
    iteration += 1
    for i in range(w.shape[0]):
        w[i, :] = compute_Wi(X.T, mu.T, 0, i)
    new_q_value = np.sum(A * w)
    if np.abs(new_q_value - q_value) <= 0.001:
        break

result = (np.array([mu[:, np.argmax(w[i, :])] for i in range(num_pixel)])).reshape(original_shape)
# plt.imshow(np.uint8(result), vmin=0, vmax=255)
plt.imsave('sunset2.png', np.uint8(result))