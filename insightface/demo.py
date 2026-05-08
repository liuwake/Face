import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis("buffalo_l")
# app = FaceAnalysis("buffalo_l", providers=["CPUExecutionProvider"])
# app = FaceAnalysis("buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0)

img = ins_get_image("t1")
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
