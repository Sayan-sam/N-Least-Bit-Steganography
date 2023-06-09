from cv2 import imread, VideoCapture
from numpy import concatenate, array
from PIL.Image import fromarray
import os

import cv2
import matplotlib.pyplot as plt

class Steganography:
    
    """
    Logic for the encoding of array with binary data
    f : Frame Pixel
    i : Image Binary
    e : Encoded Output
    f i | e
    0 0 | 0
    0 1 | 1
    1 0 | 0
    1 1 | 1

    Code logic for encode
    1. Right shift by the number of bits
    2. Left shift by the same number of bits
    3. OR operation with the Image Binary Bits

    Code logic for decode
    1. AND operation with the required number of bits
    2. Merge all bits together
    3. Separate according to the meta-data obtained
    """
    def __init__(self):
        # Global Variables
        self.bits_img_count = 4 # Number of bits for storing the number of images count
        self.bits_resolutions = [12,12,2] # Number of bits for storing the resolution of the image (height, width, channels)
        self.bits_pixel = 8 # Number of bits for storing the pixel value of our image
        self.cache_loc = "temp/temp.steg" # Location of the cache file for storing temp data
    
    def config(self, count, res, pix):
        self.bits_resolutions = res
        self.bits_pixel = pix
        self.bits_img_count = count

    def check_frame_shape(self, shape):
        if(len(shape) == 3):
            return shape
        elif(len(shape) == 2):
            return (shape[0],shape[1],1)
        else:
            raise ValueError("Shape not identified")
    
    def flatten(self,frame_list):
        return concatenate([frame.flatten() for frame in frame_list],axis=0)

    def fill_zeros(self, binary_val, length):
        if(len(binary_val)<length):
            return "0"*(length-len(binary_val))+binary_val
        elif(len(binary_val)==length):
            return binary_val
        else:
            raise ValueError("Length requested less than the Binary Length")


    def binary_conversion(self,meta_data_list, data_list):

        # res = ""
        file = open(self.cache_loc, 'w')
        count = meta_data_list[0]
        # res += self.fill_zeros(bin(count)[2:],self.bits_img_count)
        file.write(self.fill_zeros(bin(count)[2:],self.bits_img_count))
        i = 1
        while(i <= count):
            print("Meta-Data Conversion Progress: ",i,"/",count, end='\r')
            """
            res += self.fill_zeros(bin(meta_data_list[i][0])[2:], self.bits_resolutions[0])
            res += self.fill_zeros(bin(meta_data_list[i][1])[2:], self.bits_resolutions[1])
            res += self.fill_zeros(bin(meta_data_list[i][2])[2:], self.bits_resolutions[2])
            """
            file.write(self.fill_zeros(bin(meta_data_list[i][0])[2:], self.bits_resolutions[0]))
            file.write(self.fill_zeros(bin(meta_data_list[i][1])[2:], self.bits_resolutions[1]))
            file.write(self.fill_zeros(bin(meta_data_list[i][2])[2:], self.bits_resolutions[2]))
            
            i += 1
        print("Meta-Data Conversion Complete: [","="*20+"]")
        temp_percent = 0
        for i, data in enumerate(data_list):
            percent = (i*100)//len(data_list)
            if percent > temp_percent:
                self.print_progress(percent, "Data Conversion Progress:")
                temp_percent = percent
            # res += self.fill_zeros(bin(data)[2:], self.bits_pixel)
            file.write(self.fill_zeros(bin(data)[2:], self.bits_pixel))
        file.close()
        file = open(self.cache_loc, 'r')
        res = file.read()
        file.close()
        print("Data Conversion Complete: [","="*20+"]")
        if os.path.exists(self.cache_loc):
            os.remove(self.cache_loc)
        return res

    def binary_inversion(self,binary_string):
        count = int(binary_string[0:self.bits_img_count], 2)
        resolutions = []
        i = self.bits_img_count
        while(i <= count*sum(self.bits_resolutions)):
            print(" ",i,"/",count*sum(self.bits_resolutions), end='\r')
            resolutions.append(
                (int(binary_string[i:i+self.bits_resolutions[0]], 2),
                int(binary_string[i+self.bits_resolutions[0]:i+self.bits_resolutions[0]+self.bits_resolutions[1]], 2),
                int(binary_string[i+self.bits_resolutions[0]+self.bits_resolutions[1]:i+self.bits_resolutions[0]+self.bits_resolutions[1]+self.bits_resolutions[2]], 2)))
            i += sum(self.bits_resolutions)
        meta_data_list = [count] + resolutions
        print(meta_data_list)
        limit = sum([a[0]*a[1]*a[2] for a in resolutions])*self.bits_pixel+sum(self.bits_resolutions)
        print("Meta-Data Inversion Complete: [","="*20+"]")
        data_list = []
        temp_percent = 0
        for j in range(i, limit, self.bits_pixel):
            percent = (j*100)//limit
            if percent > temp_percent:
                self.print_progress(percent, "Data Inversion Progress:")
                temp_percent = percent
            data_list.append(int(binary_string[j:j+self.bits_pixel],2))
        print("Data Inversion Complete: [","="*20+"]")
        return meta_data_list, data_list
            

    def embed(self, arr, binary_string, bits):
        chunks = [binary_string[i:i+bits] for i in range(0, len(binary_string), bits)]
        temp_percent = 0
        for i,chunk in enumerate(chunks):
            percent = (i*100)//len(chunks)
            if percent > temp_percent:
                self.print_progress(percent, "Embedding Progress:")
                temp_percent = percent
            arr[i] = ((arr[i]>>bits)<<bits) | int(chunk, 2)
        print("Embedding Complete: [","="*20+"]")
        return arr

    def extract(self, arr, bits):

        # res = ""
        file = open(self.cache_loc, 'w')
        temp_percent = 0
        for i, val in enumerate(arr.tolist()):
            percent = (i*100)//len(arr)
            if percent > temp_percent:
                self.print_progress(percent, "Extracting Progress:")
                temp_percent = percent
            # res += self.fill_zeros(bin(val&((2**bits)-1))[2:],bits)
            file.write(self.fill_zeros(bin(val&((2**bits)-1))[2:],bits))
        print("Extraction Complete: [","="*20+"]")
        file.close()
        file = open(self.cache_loc, 'r')
        res = file.read()
        file.close()
        if os.path.exists(self.cache_loc):
            os.remove(self.cache_loc)
        return res
    
    def print_progress(self, percent, name = "Progress: "):
        percent = (percent//5)+1
        print(name,"["+"="*percent+">"+("-"*(20-percent))+"]",str(percent)+"/20", end='\r')

    def save_frame_list(self, image_list, folder_loc = "?", image_name = "img"):
        if(folder_loc == "?"):
            folder_loc = os.pathsep.join(self.cache_loc.split(os.pathsep)[0:-1])
        for i, image in enumerate(image_list):
            cv2.imwrite(os.path.join(folder_loc, str(image_name+"_"+str(i)+".jpg")),image)
        


    def encode(self, frame_list, image_list, bits = 1):
        # Initialization of input frames
        frame_count = len(frame_list)
        frame_shape = [self.check_frame_shape(image_frame.shape) for image_frame in frame_list]
        # frame_shape = self.check_frame_shape(frame_list[0].shape)
        pixel_arr = self.flatten(frame_list)

        # Get the parameters for the global variables | Number of Images | Resolutions of Each Image
        meta_data_list = [len(image_list)]+[self.check_frame_shape(image.shape) for image in image_list]

        # Convert the images to be encoded into a list of values
        data_list = self.flatten(image_list)

        # Calculation for the possiblilty of encoding
        data_count = len(data_list)*self.bits_pixel + (len(meta_data_list)-1)*sum(self.bits_resolutions) + self.bits_img_count
        image_count = len(pixel_arr)*bits
        print("Image count: ", image_count, "Data count: ", data_count)
        if(image_count < data_count):
            print("Encoding Not Possible!!")
            raise ValueError("Encoding Not Possible as Values are less than required")
        else: 
            print("Encoding Possible!!")
            print("Percent: ",(data_count/image_count)*100)

        # Converting the list of values into binary string
        binary_string = self.binary_conversion(meta_data_list, data_list)
        
        # Embed the binary
        pixel_arr = self.embed(pixel_arr, binary_string, bits)

        # Reconstruct the input frames
        frame_list = []
        for i in range(0,frame_count):
            total_pixel = frame_shape[i][0]*frame_shape[i][1]*frame_shape[i][2]
            temp_arr = pixel_arr[0:total_pixel]
            frame_list.append(temp_arr.reshape(frame_shape[i]))
            pixel_arr = pixel_arr[total_pixel:]
        
        return frame_list
    
    def decode(self, frame_list, bits = 1):
        # Initialization of the input frame
        pixel_arr = self.flatten(frame_list)

        # Extract all the values needed from the pixel array
        binary_string = self.extract(pixel_arr, bits)
        meta_data_list, data_list = self.binary_inversion(binary_string)
        
        # Form the image list from the meta-data
        image_list = []
        image_count = meta_data_list[0]
        i = 1
        while(i <= image_count):
            resolution_shape = meta_data_list[i]
            resolution_product = resolution_shape[0]*resolution_shape[1]*resolution_shape[2] # type: ignore
            image_arr = array(data_list[0:resolution_product])
            data_list = data_list[resolution_product:]
            image_list.append(image_arr.reshape(resolution_shape))
            i = i+1
        return image_list

    def encode_image(self, img_loc:str, en_img_loc:str, bits = 2):
        img = imread(img_loc)
        en_img = imread(en_img_loc)
        return self.encode([img], [en_img], bits)[0]
    
    def decode_image(self, img_loc:str, bits = 2):
        img = imread(img_loc)
        return self.decode([img], bits)[0]
    
    def encode_video(self, video_loc:str, img_dir:str, bits = 2):
        # Preprare the video to list of frames
        capture = VideoCapture(video_loc, apiPreference=cv2.CAP_MSMF)
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        print("Frame rate: ",frame_rate)
        image_list = []
        while (True):
            ret, frame = capture.read()
            if ret:
                image_list.append(frame)
            else:
                break
        shape = image_list[0].shape
        print("Image shape: ",shape)

        # Prepare the image directory into a list of frames
        dir_list = [os.path.join(img_dir, loc) for loc in os.listdir(img_dir)]
        en_img_list = [imread(path) for path in dir_list]

        # Encode the images and form the video
        image_list = self.encode(image_list, en_img_list, bits = bits)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(str(video_loc.split(".")[0]+str("_encoded.mp4")),fourcc, frame_rate, (shape[1],shape[0]))

        for image in image_list:
            video_output.write(image)
        
        cv2.destroyAllWindows()
        video_output.release()
        return str(video_loc.split(".")[0]+str("_encoded.mp4"))


    
    def decode_video(self, video_loc:str, bits = 2):
        capture = VideoCapture(video_loc)
        image_list = []
        while (True):
            ret, frame = capture.read()
            if ret:
                image_list.append(frame)
            else:
                break
        
        return self.decode(image_list, bits)

    def encode_image_dir(self, img_dir, en_dir, bits = 2):
        # Prepare the image directory into a list of frames
        dir_list = [os.path.join(img_dir, loc) for loc in os.listdir(img_dir)]
        img_list = [imread(path) for path in dir_list]
        dir_list = [os.path.join(en_dir, loc) for loc in os.listdir(en_dir)]
        en_list = [imread(path) for path in dir_list]

        result = self.encode(img_list, en_list, bits)
        self.save_frame_list(result, "images/output", "output")
        return "images/output"
    
    def decode_image_dir(self, img_dir, bits):
        # Prepare the image directory into a list of frames
        dir_list = [os.path.join(img_dir, loc) for loc in os.listdir(img_dir)]
        img_list = [imread(path) for path in dir_list]
        result = self.decode(img_list, bits)
        self.save_frame_list(result, "images/output_secret", "secret")
        return "images/output_secret"
    

    
    


    
