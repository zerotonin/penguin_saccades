from tqdm import tqdm
import pandas as pd
import os
from shapely.geometry import Point, Polygon
import numpy as np
class read_charon_tra():
    def __init__(self,file_position,indice_file = '',roi = []):

        self.file_position = file_position # filePosition = file name
        self.rawTextData  = []           # preallocation with empty list
        self.indice_file = indice_file
        self.roi = Polygon(roi)

    def read_raw_text_entire_file(self):
        temporay_file_dialog = open(self.file_position, 'r')  # 'r': open for reading
        self.rawTextData = temporay_file_dialog.readlines()  # reading row wise 
        temporay_file_dialog.close()
        #print(self.rawTextData)               # and print it into the rawTextData List?

    def split_line_into_image_objects(self,line): 
        '''
        This function splits all objects in a line of raw text data. 
        The strings are split by the > symbol
        Function returns a list of string containing all found objects in this line, but ignores the framenumber.
        '''
        object_list = line.split('>')  # each object is encapsuilated in ><
        del(object_list[0]) # first entry is frameNumber
        return object_list

    def get_frame_number_from_line(self,line):
        '''
        This function returns the frame number as int from a given text data line.
        '''
        object_list = line.split('>')
        frame_number = int(object_list[0][0:-3])
        return frame_number

    def split_image_object_to_list(self,image_object_string):
        '''
        Each object consist out of one string (the object name) and 5 numbers: quality ,y0,x0,y1,x1
        The last for are the minimum and maximum of the bounding box of the object given
        in coordinates normalised to the picture size.
        The string containing this data is seperated by commata and is now split at these and the 
        numerical data are converted into floats. The return value is a list consistent of a string
        and five floats. 
        '''
        object_value_list=image_object_string.split(',') # object Values are seperated by ','
        object_value_list[5]= object_value_list[5][0:-2] # delete the last two characters of the string ' <'
        # convert all values into float except of the first which is the name of the object already given as a string
        for i in range(1,6):
            object_value_list[i]= float(object_value_list[i]) # convert string into floating point number
        # return object values in list
        return object_value_list

    def image_object_list_to_dict(self,object_value_list): # putting objectValues into Dictionary
        '''
        This function transforms the list of object values returned by self.readImageObject
        into a dictionary with the following keys and values

        keys         : values
        =============:======================================================================
        name         : a string with the identifier of the object, e.g. 'fly','arena','marker'
        centerOfMass : a tuple holding two floats the x and y coordinate normalised to the 
                       image size 
        quality      : a float with the normalized quality of detection 0->1
        boundingBox  : a tuple of four float with the minimum and maximum coordinates of the 
                       bounding box in the succession (y0,x0,y1,x1)
        '''
        # shorthands for name and quality
        image_objec_name = object_value_list[0]
        quality   = object_value_list[1]
        # combine coordinates into a tuple to avoid permutation of coordinates
        bounding_box_coordinates = tuple(object_value_list[2::])   # a Tupel is a finite ordered list of elements
        # caclulate center of mass
        y,x = self.calculate_center_of_mass_from_BB(bounding_box_coordinates)
        center_of_mass = (y,x) # make tuple to avoid permutation

        # define a dictionary with the data and return it
        return {'name':image_objec_name,'centerOfMass': center_of_mass, 'quality': quality, 'boundingBox': bounding_box_coordinates} #dictionary{key:value pairs}

    def calculate_center_of_mass_from_BB(self,bounding_box_coordinates):
        '''
        To get the middle or center of mass of a bounding box you need to
        calculate the mean of the x-coordinates and y-coordinates respectively.
        '''
        y = (bounding_box_coordinates[0]+bounding_box_coordinates[2])/2.0
        x = (bounding_box_coordinates[1]+bounding_box_coordinates[3])/2.0
        return y,x
    
    def read_image_object_per_line(self,line): 
        # get image objects as list of strings from line
        image_objects  = self.split_line_into_image_objects(line)  
        # get frame number as integer from line
        frame_number   = self.get_frame_number_from_line(line)
        # initialize return list with frame number
        image_object_dict_list = [frame_number]
        # this for loop transverses through the list of string img Obj
        for image_object in image_objects:
            # object string is split in different values and numericals are converted to floats
            object_value_list = self.split_image_object_to_list(image_object)
            # converts list of data into a structer dict
            image_object_dictionary = self.image_object_list_to_dict(object_value_list)
            # add dict to return list
            if self.test_if_animal_in_roi(image_object_dictionary['centerOfMass']):
                image_object_dict_list.append(image_object_dictionary)       
        return image_object_dict_list

    def convert_entire_raw_file_to_dicts(self):
        '''
        Convert the raw text data into a list in which each element is the data of a frame.
        Each frame consists of a list, which first element is the frame number as an integer. All following
        elements are dictionaries in the following format:

        keys         : values
        =============:======================================================================
        name         : a string with the identifier of the object, e.g. 'fly','arena','marker'
        centerOfMass : a tuple holding two floats the x and y coordinate normalised to the 
                       image size 
        quality      : a float with the normalized quality of detection 0->1
        boundingBox  : a tuple of four float with the minimum and maximum coordinates of the 
                       bounding box in the succession (ymin,xmin,ymax,ymin)

        '''
        # preallocation of an empty list
        self.image_object_data = list()
        # for loop transverses each line of the file
        for line in tqdm(self.rawTextData,desc = 'converting text to dict'):
            # each line is read and converted into an image object dictionary list and appended to self.imgObjData
            self.image_object_data.append(self.read_image_object_per_line(line))

    def read_entire_small_file(self):
        '''
        Main reader class.
        Reads the file at the position stored in self.filePosition.
        The raw text data is stored in a list of strings in self.rawTextData.
        The output list of image object dictionaries is stored in self.imObjData.
        '''
        # read text data from file
        self.read_raw_text_entire_file()
        # convert text data 2 img object dictionaries
        self.convert_entire_raw_file_to_dicts()

    def read_file(self,start_line = 0, maximum_lines = -1, show_progress = True):
        #self.image_object_data = list()
        file_dialog = open(self.file_position, 'r')
        line_count  = 0
        progress_bar_str='-\|/' 
        while (maximum_lines == -1) or (line_count <= maximum_lines):
            #update counter
            line_count += 1
            # Get next line from file
            line = file_dialog.readline()
            # if line is empty end of file is reached
            if not line:
                break
            
            if line_count >= start_line:            
                #update progress bar
                if show_progress:
                    os.system("printf '\033c'")
                    print(f'{progress_bar_str[line_count%4]} reading line {line_count}', flush=True)
                #s data
                self.image_object_data.append(self.read_image_object_per_line(line))
 
        file_dialog.close()

    def test_if_animal_in_roi(self,center_of_mass):
        # test if there is a roi = polygon
        if self.roi:
            point_of_mass = Point(center_of_mass)
            return point_of_mass.within(self.roi)
        # if no roi is defined return that the animal is in it as it only needs to be detected
        else:
            return True
        
    def delete_empty_frames(self):
        self.image_object_data = [inner_list for inner_list in self.image_object_data if len(inner_list)> 1]
        
    def chose_best_detection(self):
        for i in range(len(self.image_object_data)):
            frame = self.image_object_data[i]
            if len(frame)>2:
                # Could be that all obkects are sorted by their quality and this might be superfluous
                qualities = [img_object['quality'] for img_object in frame[1::] ]
                max_index = np.argmax(qualities)
                frame =  [frame[0],frame[max_index+1]]
                self.image_object_data[i] = frame

    def only_read_specific_lines_from_tra_file(self):
        mp4_file_name = os.path.basename(self.file_position)[:-3]+"mp4"
        data = pd.read_csv(self.indice_file)
        mp4_file_data = data[data['filename'] == mp4_file_name]
        for i,x in mp4_file_data.iterrows():
            self.image_object_data = list()
            self.read_file(x.start, x.end-x.start) 
            self.delete_empty_frames()
            self.chose_best_detection()
            df = pd.DataFrame([frame[1] for frame in self.image_object_data],
                              index= [frame[0] for frame in self.image_object_data])
            df['center_of_mass_x'] = df['centerOfMass'].apply(lambda x: x[1])
            df['center_of_mass_y'] = df['centerOfMass'].apply(lambda x: x[0])
            df['bounding_box_x0']  = df['boundingBox'].apply(lambda x: x[1])
            df['bounding_box_y0']  = df['boundingBox'].apply(lambda x: x[0])
            df['bounding_box_x1']  = df['boundingBox'].apply(lambda x: x[3])
            df['bounding_box_y1']  = df['boundingBox'].apply(lambda x: x[2])
            df = df.drop('centerOfMass', axis=1)
            df = df.drop('boundingBox', axis=1)

            return df
            #df.to_hdf("iterierenderName.h5", key='trace')



        #self.read_file(mp4_file_data['start'][0],mp4_file_data['end'][0])
        

if __name__ == '__main__':
    source_file = '/path/to/penguins/Rockhopper_05-03-2021.tra' 
    indice_file = '/path/to/penguins/Penguin_video_data.csv'
    #roi in pixel coordinates
    roi = np.array([(0,0),(660,0),(660,272),(1402,272),(1402,788),(0,788)],dtype=float)
    # roi in rel. coordinates
    roi[:,0]=roi[:,0]/1402.
    roi[:,1]=roi[:,1]/788.
    # roi in tensorflow coordinates
    roi = np.fliplr(roi)
    reader = read_charon_tra(source_file,indice_file,roi)
    df = reader.only_read_specific_lines_from_tra_file()
    
    
    # Reindex the DataFrame with a complete range of indices
    #min_idx, max_idx = df.index.min(), df.index.max()
    #df_reindexed = df.reindex(range(min_idx, max_idx + 1))
    #df_interpolated = df_reindexed.interpolate(method='polynomial', order=2)
    # import matplotlib.pyplot as plt   
    #df_interpolated.plot('center_of_mass_x','center_of_mass_y')
    #plt.show()
 