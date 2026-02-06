import  numpy as np
N = 64

if __name__ == "__main__":
    res = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            res[i][j] = 2 / N**2 * np.cos(2*np.pi / N * (5*i + 5*j))

    for i in range(N):
        print(res[i][N-1-i])