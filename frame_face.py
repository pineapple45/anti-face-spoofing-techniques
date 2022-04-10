import cv2
import os
import tensorflow as tf
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  

def extract_face(img):
    # Convert to grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  
    
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        # cv2.imshow("face",faces)
        # cv2.imwrite('face.jpg', faces)
        
    # Display the output
    # cv2.imwrite('detcted.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    return faces

# def extractImages(pathIn, pathOut,frame_rate=500):
#     count = 0
#     vidcap = cv2.VideoCapture(pathIn)
#     success,image = vidcap.read()
#     while True:
#         vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*frame_rate))    # added this line 
#         success,image = vidcap.read()
#         # if success == False:
#         #     break
#         print ('Read a new frame: ', success)
#         cropped_faces = extract_face(image)
#         cv2.imwrite( pathOut + "frame%d.jpg" % count, cropped_faces)     # save frame as JPEG file
#         count = count + 1
    
#     vidcap.release()


# extractImages('./test.mov','./data/train/real/')

# START CAPTURING VIDEOS

def extract_multiple_videos(intput_filenames,pathOut,frame_rate, resize):
    """Extract video files into sequence of images.
       Intput_filenames is a list for video file names"""

    i = 0  # Counter of first video

    # Iterate file names:
    for intput_filename in intput_filenames:
        sec_count = 0
        cap = cv2.VideoCapture(intput_filename)
        cap.set(cv2.CAP_PROP_POS_MSEC,(sec_count*frame_rate))

        # Keep iterating break
        while True:
            ret, frame = cap.read()  # Read frame from first video

            if ret:
                try:
                    cropped_faces = extract_face(frame)
                    cropped_faces = cv2.resize(cropped_faces,(resize,resize))
                    # cropped_faces = Image.fromarray(cropped_faces.astype('uint8'))
                    # cropped_faces = cropped_faces.resize((resize,resize))
                    # cropped_faces = np.array(cropped_faces)
                    cv2.imwrite(pathOut + f'frame_hand_{i}.jpg', cropped_faces)  # Write frame to JPEG file (1.jpg, 2.jpg, ...)
                    # cv2.imshow('frame', frame)  # Display frame for testing
                    i += 1 # Advance file counter
                    sec_count += 1
                except:
                    pass
            else:
                # Break the interal loop when res status is False.
                break

            # cv2.waitKey(100) #Wait 100msec (for debugging)

        cap.release() #Release must be inside the outer loop
    return i


def start(in_dir , out_dir,frame_rate=500,resize=256):
    # traverse whole directory
    dir_list = []
    for root, dirs, files in os.walk(f'{in_dir}'):
        # select file name
        for file in files:
            # check the extension of files
            if file.endswith('.mov'):
                # print whole path of files
                path = os.path.join(root, file)
                # print(path)
                dir_list.append(path)
                # extractImages(path,'./data/train/real',frame_rate=15000)

    count = extract_multiple_videos(dir_list,f'{out_dir}',frame_rate=frame_rate, resize=resize)
    print(f'TOTAL FRAMES/IMAGES FORMED: {count}')


# train real files

train_dir_real = {"in_dir": './replayattack/train_files/train/real',"out_dir":'./data/train/real/'}
# train_dir_real = {"in_dir": './test',"out_dir":'./test/'}

# train attack files

train_dir_attack_fixed = {"in_dir":'./replayattack/train_files/train/attack/fixed', 'out_dir':'./data/train/attack/'}
train_dir_attack_hand = {"in_dir":'./replayattack/train_files/train/attack/hand', 'out_dir':'./data/train/attack/'}

# test real files

test_dir_real = {"in_dir": './replayattack/test_files/test/real',"out_dir":'./data/test/real/'}


# test attack files

test_dir_attack_fixed = {"in_dir":'./replayattack/test_files/test/attack/fixed', 'out_dir':'./data/test/attack/'}
test_dir_attack_hand = {"in_dir":'./replayattack/test_files/test/attack/hand', 'out_dir':'./data/test/attack/'}


# val real files

val_dir_real = {"in_dir": './replayattack/dev_files/devel/real',"out_dir":'./data/val/real/'}


# val attack files

val_dir_attack_fixed = {"in_dir":'./replayattack/dev_files/devel/attack/fixed', 'out_dir':'./data/val/attack/'}
val_dir_attack_hand = {"in_dir":'./replayattack/dev_files/devel/attack/hand', 'out_dir':'./data/val/attack/'}

'''
10 FPS
1 video = 15 secs
15 secs = 15 * 10 = 150 frames + 1(extra) frame = 151 frames
60 videos total
total frames = 60 * 151 = 9060 frames(images)

BATCH_SIZE = 4 --> FRAMES per BATCH = 2265 (9060/4) --> DROP = 0 (9060%4) frames
BATCH_SIZE = 32 --> FRAMES per BATCH = 283 (9060/32) --> DROP = 4 (9060%32) frames
BATCH_SIZE = 64 --> FRAMES per BATCH = 141 (9060/64) --> DROP = 36 (9060%64) frames
'''
# BATCH SIZES TO CATER - 4 , 32 , 64


# start(train_dir_real['in_dir'],train_dir_real['out_dir'],frame_rate=100,resize=256) 
# --- TOTAL FRAMES/IMAGES FORMED: 22490

# start(train_dir_attack_fixed['in_dir'],train_dir_attack_fixed['out_dir'],frame_rate=100,resize=256) 
# --- TOTAL FRAMES/IMAGES FORMED: 35466

# start(train_dir_attack_hand['in_dir'],train_dir_attack_hand['out_dir'],frame_rate=100,resize=256) 
# --- TOTAL FRAMES/IMAGES FORMED: 33749

# start(test_dir_real['in_dir'],test_dir_real['out_dir'],frame_rate=100,resize=256) 
# --- TOTAL FRAMES/IMAGES FORMED: 29229

# start(test_dir_attack_fixed['in_dir'],test_dir_attack_fixed['out_dir'],frame_rate=100,resize=256) 
# --- TOTAL FRAMES/IMAGES FORMED: 47125

# start(test_dir_attack_hand['in_dir'],test_dir_attack_hand['out_dir'],frame_rate=100,resize=256) 
# --- TOTAL FRAMES/IMAGES FORMED: 44674

# start(val_dir_real['in_dir'],val_dir_real['out_dir'],frame_rate=100,resize=256)
# --- TOTAL FRAMES/IMAGES FORMED: 22485

# start(val_dir_attack_fixed['in_dir'],val_dir_attack_fixed['out_dir'],frame_rate=100,resize=256)
# --- TOTAL FRAMES/IMAGES FORMED: 34738

# start(val_dir_attack_hand['in_dir'],val_dir_attack_hand['out_dir'],frame_rate=100,resize=256)
# --- TOTAL FRAMES/IMAGES FORMED: 33729







