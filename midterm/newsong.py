import serial 
import time

song = [
  349, 440, 523, 587, 523, 523, 349, 392, 392, 349, 440, 440,
  440, 523, 587, 659, 587, 523, 440, 392, 392, 349, 392, 392, 440, 349, 349,
  349, 440, 523, 587, 523, 523, 349, 392, 392, 349, 440, 440,
  440, 523, 587, 698, 784, 880, 784, 698, 587, 698, 784, 784, 698, 698, 698,
  587, 698, 784, 784, 698, 698, 698, 0]

noteLength = [
  1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 2, 1]
  
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)

print("Sending signal...")

s.write(bytes("SeeYouAgain\n", 'UTF-8'))

for i in song:
    s.write(bytes("%d\n" % i, 'UTF-8'))
for i in noteLength:
    s.write(bytes("%d\n" % i, 'UTF-8'))

s.close()
print("Signal sending complete!")