from subprocess import call

call(["fswebcam","-d","/dev/video0","-S","2","-s","brightness=60%","-s","Contrast=20%","-s","Gamma=50%","-r","480x480","-s","Sharpness=40%","test.jpg"])
