import cv2
from src import config
from src.detector import FaceFeatureExtractor

cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

with FaceFeatureExtractor() as extractor:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from camera")
            break
        frame, ear, mar = extractor.process_frame(frame)
        text = "no face" if ear is None else f"EAR {ear:.2f}  MAR {mar:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imshow("detector test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()