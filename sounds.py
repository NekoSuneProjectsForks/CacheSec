### USE THESE FUNCTIONS IN YOUR CODE TO PLAY SOUNDS WHEN ACCESS IS GRANTED OR DENIED ###




from gpiozero import PWMOutputDevice
from time import sleep

speaker = PWMOutputDevice(18)

def tone(freq, duration, volume=0.5, gap=0.02):
    speaker.frequency = freq
    speaker.value = volume
    sleep(duration)
    speaker.off()
    sleep(gap)

# ✅ ACCESS GRANTED
def access_granted():
    print("ACCESS GRANTED")
    tone(1000, 0.05)
    tone(1400, 0.05)
    tone(1800, 0.08)
    sleep(0.05)
    tone(2200, 0.12)

# ❌ ACCESS DENIED
def access_denied():
    print("ACCESS DENIED")
    tone(400, 0.15)
    tone(300, 0.15)
    tone(200, 0.25)
    sleep(0.05)
    # buzzy error tone
    for _ in range(3):
        tone(250, 0.05)
        sleep(0.03)

try:
    access_granted()
    sleep(1)
    access_denied()

finally:
    speaker.off()
    speaker.close()