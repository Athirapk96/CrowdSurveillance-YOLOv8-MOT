#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2


# In[2]:


class ImageIterator:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(self.folder_path)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.current_index < len(self.file_list):
            file_name = self.file_list[self.current_index]
            file_path = os.path.join(self.folder_path, file_name)

            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                self.current_index += 1
                return image
            else:
                self.current_index += 1

        raise StopIteration


