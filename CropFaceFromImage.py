import cv2


class CropFace:

    def __init__(self, imagepath):
        self.imagepath = imagepath

    def cropimages(self):
        image = cv2.imread(self.imagepath, 0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.3, minNeighbors=5)

        faces_found = []

        for x, y, w, h in faces:
            facecrop = image[y:y + h, x:x + w]
            faces_found.append(facecrop)

        return faces_found

    def saveimages(self, facesarray):
        if len(facesarray) == 0:
            print("No faces found, try another image please")

        for i, face in enumerate(facesarray):
            cv2.imwrite("face-" + str(i) + ".jpg", face)
            print("Saved image as face-" + str(i))


if __name__ == "__main__":
    crop = CropFace("img1.jpg")
    faces = crop.cropimages()
    crop.saveimages(faces)
