import cv2
import os

#go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
entries = sorted(os.listdir('icdar2017-training-color'))

# empty list to store the images files names
files_list = []
# the directory containing the image files

for entry in entries:
    #print(entry)
    files_list.append(entry)

string = 'icdar2017-training-color'
string2 = 'patches_images'

def make_patches(image):
    sample = cv2.imread(os.path.join(string, image))

    height = sample.shape[0]
    width = sample.shape[1]

    #cut the image in half
    width_cutoff = width//2
    left1 = sample[:, :width_cutoff]
    right1 = sample[:, width_cutoff:]
    #finish vertical devide image
    ##########################################
    # At first Horizontal devide left1 image #
    ##########################################
    #rotate image LEFT1 to 90 CLOCKWISE
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    l1 = img[:, :width_cutoff]
    l2 = img[:, width_cutoff:]
    # finish vertical devide image
    #rotate image to 90 COUNTERCLOCKWISE
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    cv2.imwrite(os.path.join(string2, image[:-4] + '_1' + '.jpg'), l1)
    #rotate image to 90 COUNTERCLOCKWISE
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    cv2.imwrite(os.path.join(string2, image[:-4] + '_2' + '.jpg'), l2)
    ##########################################
    # At first Horizontal devide right1 image#
    ##########################################
    #rotate image RIGHT1 to 90 CLOCKWISE
    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    r1 = img[:, :width_cutoff]
    r2 = img[:, width_cutoff:]
    # finish vertical devide image
    #rotate image to 90 COUNTERCLOCKWISE
    r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    cv2.imwrite(os.path.join(string2, image[:-4] + '_3' + '.jpg'), r1)
    #rotate image to 90 COUNTERCLOCKWISE
    r2 = cv2.rotate(r2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    cv2.imwrite(os.path.join(string2, image[:-4] + '_4' + '.jpg'), r2)


for i in range(len(files_list)):
    make_patches(files_list[i])

