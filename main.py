import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image_path = "img.png"
original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
displayed = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Étalonnage de l'image
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
calibrated_image = clahe.apply(image)

# Binarisation de l'image 
_, binary_image = cv2.threshold(calibrated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Détection des contours avec différents algorithmes
edges1 = cv2.Canny(binary_image, 30, 100)
sobelx = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0,ksize=3)
sobely = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0,ksize=3)
edges2 = np.sqrt(sobelx**2 + sobely**2)
laplace = cv2.Laplacian(binary_image, cv2.CV_64F)
edges3 = np.uint8(laplace)

# Application de traitements d'image
kernel = np.ones((5,5), np.uint8)
erode = cv2.erode(binary_image,kernel,iterations=1)
dilat = cv2.dilate(binary_image, kernel, iterations=1)
blur = cv2.blur(calibrated_image,(11,11))
Gausblur = cv2.GaussianBlur(calibrated_image,(11,11),0)
highcontrast = cv2.convertScaleAbs(calibrated_image,alpha=0.2,beta=0)
lowcontrast = cv2.convertScaleAbs(calibrated_image,alpha=5,beta=0)
highlum = cv2.convertScaleAbs(calibrated_image,alpha=1,beta=100)
lowlum = cv2.convertScaleAbs(calibrated_image,alpha=1,beta=-100)
corners = cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)
cornered = displayed.copy()
for i in corners:
    x, y = i.ravel()
    cv2.circle(cornered, (x, y), 10, 255, -1)
noise = np.random.normal(0,50,calibrated_image.shape)
bruite = np.clip(calibrated_image+noise,0,255).astype(np.uint8)
debruite = cv2.GaussianBlur(bruite,(11,11),0)
highres = cv2.addWeighted(Gausblur,1.5,Gausblur,-0.5,0)


# Affichage des résultats
plt.subplot(5, 4, 1), plt.imshow(displayed, cmap='gray')
plt.title('Image Originale'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 2), plt.imshow(calibrated_image, cmap='gray')
plt.title('Image Étalonnée'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 3), plt.imshow(binary_image, cmap='gray')
plt.title('Image Binarisée'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 4), plt.imshow(edges1, cmap='gray')
plt.title('Contours (Canny)'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 5), plt.imshow(edges2, cmap='gray')
plt.title('Contours (Sobel)'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 6), plt.imshow(edges3, cmap='gray')
plt.title('Contours (Laplacien)'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 7), plt.imshow(erode, cmap='gray')
plt.title('Erosion'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 8), plt.imshow(dilat, cmap='gray')
plt.title('Dilatation'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 9), plt.imshow(blur, cmap='gray')
plt.title('Flou'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 9), plt.imshow(blur, cmap='gray')
plt.title('Flou'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 10), plt.imshow(Gausblur, cmap='gray')
plt.title('Flou Gaussien'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 11), plt.imshow(highcontrast, cmap='gray')
plt.title('Haut contraste'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 12), plt.imshow(lowcontrast, cmap='gray')
plt.title('Faible contraste'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 13), plt.imshow(highlum, cmap='gray')
plt.title('Haute luminosité'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 14), plt.imshow(lowlum, cmap='gray')
plt.title('Faible luminosité'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 15), plt.imshow(cornered, cmap='gray')
plt.title('Point d\'intéret'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 16), plt.imshow(bruite, cmap='gray')
plt.title('Bruité'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 17), plt.imshow(debruite, cmap='gray')
plt.title('Débruité'), plt.xticks([]), plt.yticks([])

plt.subplot(5, 4, 18), plt.imshow(highres, cmap='gray')
plt.title('Haute résolution'), plt.xticks([]), plt.yticks([])


plt.show()
