import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.9)

camera = jetson.utils.videoSource("/home/nvidia/jetson-inference/data/images/dog_0.jpg")

display = jetson.utils.videoOutput("dog_2_result.jpg")

while display.IsStreaming():
    img = camera.Capture()
    if img is None:
       break

    detections = net.Detect(img)

    print(f"Total objects detected: {len(detections)}")
    for detection in detections:
        print(f"ClassID: {detection.ClassID}")
        print(f"Confidence: {detection.Confidence}")
        print(f"Left: {detection.Left}, Top: {detection.Top}")
        print(f"Right: {detection.Right}, Bottom: {detection.Bottom}")
        print(f"Width: {detection.Width}, Height: {detection.Height}")
        print(f"Area: {detection.Area}, Center: {detection.Center}")
        print("---")

    display.Render(img)
    break
