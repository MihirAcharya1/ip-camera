import os
import cv2

# Force FFMPEG backend to use strict TCP and completely ignore TLS verification drops
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|tls_verify;0"

# Use the dedicated stream profile parameters (No special characters in password)
username = "user"
password = "india$Me123"
ip_address = "192.168.31.241"
port = "554"

# Use 'rtsps' instead of 'rtsp' to accommodate the camera's active TLS 1.2/1.3 layer
secure_rtsp_url = f"rtsps://{username}:{password}@{ip_address}:{port}/video/live?channel=1&subtype=1"

print(f"Connecting to encrypted camera pipeline: {secure_rtsp_url}")
cap = cv2.VideoCapture(secure_rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("\n[Secure Path 1 Failed] Attempting core architecture secure fallback...")
    # Alternative Dahua architecture secure stream path query
    secure_rtsp_url_alt = f"rtsps://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=1"
    cap = cv2.VideoCapture(secure_rtsp_url_alt, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("\n[-] Error: The connection was dropped by the camera firewall.")
    print("Please confirm that 'user' has been added to the camera Account settings.")
    exit()

print("[+] Stream Connection Established Successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Dropped video packet frames.")
        break

    cv2.imshow("CP Plus Secure Live Feed", frame)

    # Press the 'q' key to terminate the streaming loop window cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
