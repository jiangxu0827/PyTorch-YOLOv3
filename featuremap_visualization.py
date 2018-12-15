import cv2

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)

image = test_images[5].data
image = image.cpu().numpy()
print(image.shape)
image = np.transpose(image, (1,2,0))
plt.imshow(np.squeeze(image), cmap='gray')

plt.subplot(1,2,2)
weights = net.conv1.weight.data
w = weights.cpu().numpy()
c = cv2.filter2D(image, -1, w[2][0])
plt.imshow(c, cmap='gray')

print(w.shape)
print(w[0][0])