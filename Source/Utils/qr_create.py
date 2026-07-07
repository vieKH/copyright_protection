import qrcode
import numpy as np
import matplotlib.pyplot as plt
# Dữ liệu cần mã hóa
data = "https://coruscating-lebkuchen-901ade.netlify.app/"

# Tạo QR
qr = qrcode.QRCode(
    version=1,          # QR Version 1 (21x21)
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0            # Không viền để dễ xem ma trận
)

qr.add_data(data)
qr.make(fit=True)

# True = ô đen, False = ô trắng
matrix = np.array(qr.get_matrix(), dtype=int)

# Đổi về:
# 1 = đen
# 0 = trắng

matrix = 1 - matrix


h, w = matrix.shape

print(f"Kích thước QR: {w} x {h}")
print()

print(matrix)
plt.figure()
plt.imshow(matrix, cmap="gray")
plt.show()