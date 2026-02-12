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
    sum = np.zeros((32 //2 - L, 32//2 - L))
    print(sum.shape[0], sum.shape[1])