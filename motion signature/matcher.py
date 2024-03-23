import cv2 as cv 
import numpy as np 
from ffpyplayer.player import MediaPlayer
import pyglet
import sys
  



# READING THE MOTION SIGNATURE NUMPY FILES THAT IS ALREADY CREATED
v1 = np.load('motion signature/video1.npy') 
v2 = np.load('motion signature/video2.npy') 
v3 = np.load('motion signature/video3.npy') 
v4 = np.load('motion signature/video4.npy') 
v5 = np.load('motion signature/video5.npy') 
v6 = np.load('motion signature/video6.npy') 
v7 = np.load('motion signature/video7.npy') 
v8 = np.load('motion signature/video8.npy') 
v9 = np.load('motion signature/video9.npy') 
v10 = np.load('motion signature/video10.npy') 
v11 = np.load('motion signature/video11.npy') 
v12 = np.load('motion signature/video12.npy') 
v13 = np.load('motion signature/video13.npy') 
v14 = np.load('motion signature/video14.npy') 
v15 = np.load('motion signature/video15.npy') 
v16 = np.load('motion signature/video16.npy') 
v17 = np.load('motion signature/video17.npy') 
v18 = np.load('motion signature/video18.npy') 
v19 = np.load('motion signature/video19.npy') 
v20 = np.load('motion signature/video20.npy') 

# HASHMAP TO SAVE THE VIDEOS
v = {
    "v1": v1,
    "v2": v2,
    "v3": v3,
    "v4": v4,
    "v5": v5,
    "v6": v6,
    "v7": v7,
    "v8": v8,
    "v9": v9,
    "v10": v10,
    "v11": v11,
    "v12": v12,
    "v13": v13,
    "v14": v14,
    "v15": v15,
    "v16": v16,
    "v17": v17,
    "v18": v18,
    "v19": v19,
    "v20": v20,
}


# WHICH QUERY VIDEO WE ARE TRYING TO FIND
#videoNum = "8"
#cap = cv.VideoCapture("Queries/video" + videoNum  + "_1.mp4") 
cap = cv.VideoCapture(sys.argv[1]) 

ret, first_frame = cap.read() 
  
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) 
  
mask = np.zeros_like(first_frame) 
  
mask[..., 1] = 255

arr = np.array([])
result = 0
while(cap.isOpened()): 
    ret, frame = cap.read() 

    if (not ret):
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,  
                                    None, 
                                    0.5, 3, 15, 3, 5, 1.2, 0) 
    
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) 

    arr = np.append(arr, np.mean(magnitude))

    count = 0
    idx = 0
    for x in range(1,21):
        k = "v" + str(x)
        intersection = np.in1d(arr, v[k])
        if not(False in intersection):
            count += 1
            idx = x

        if (count > 1):
            break

    if (count == 1):
        result = idx
        break

    prev_gray = gray 
    
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
  
# STARTING FRAME OF MATCHED VIDEO
starting_frame = 0
l = np.size(arr)
vid = v["v" + str(result)]
end = np.size(vid) - l
for x in range(0, end):
    if (arr == vid[x:x+l]):
        starting_frame = x
        break

#print(starting_frame)
#print(result)

video_path="video/video" + str(result) + ".mp4"



def play_video_from_time(video_path, start_time):
    window = pyglet.window.Window(352, 288)

    player = pyglet.media.Player()
    source = pyglet.media.StreamingSource()
    media = pyglet.media.load(video_path)

    player.queue(media)
    player.seek(start_time)
    player.play()

    @window.event
    def on_draw():
        window.clear()
        if player.source and player.source.video_format:
            player.get_texture().blit(0, 0)

    @window.event 
    def on_key_press(symbol, modifier): 
    
        # key "p" get press 
        if symbol == pyglet.window.key.P: 
            
            # pause the video
            player.pause()
            
            # printing message
            print("Video is paused")
            
            
        # key "r" get press 
        if symbol == pyglet.window.key.R: 
            
            # resume the video
            player.play()
            
            # printing message
            print("Video is resumed")
        
        # key "t" for reset
        if symbol == pyglet.window.key.T: 
            player.seek(0)
            print("Reset Video")

    pyglet.app.run()

# STARTING FRAME/30FPS = STARTING TIME IN SECOND
start_time_seconds = starting_frame * 1.0 / 30.0  # Starting time in seconds

play_video_from_time(video_path, start_time_seconds)





cap.release() 
cv.destroyAllWindows() 




