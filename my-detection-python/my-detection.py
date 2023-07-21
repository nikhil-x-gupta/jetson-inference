#!/usr/bin/env python3

import jetson.inference
import jetson.utils

thresh = 0.5
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=thresh)

camera = jetson.utils.videoSource("/dev/video0")

display = jetson.utils.videoOutput("my_video.mp4")

while True:
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
