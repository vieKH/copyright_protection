import  numpy as np
import matplotlib.pyplot as plt
L = 8
if __name__ == "__main__":

    rng = np.random.default_rng(0)
    qr = (rng.random((L, L)) > 0.5).astype(np.uint8)
    # plt.figure(figsize=(15,8))
    # plt.imshow(qr, cmap="gray")
    # plt.axis("off")
    # plt.show()
    sum = np.array([1,2,3,4,5])

    print(sum.shape[0])