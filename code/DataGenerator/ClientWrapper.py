from io import BytesIO as BytesIO
import numpy as np
import random
import cv2 as cv
import PIL.Image as Image
import pdb
import os
import time


class WrappedClient(object):
    def __init__(self, client, DataStoragePath, HighResFactor, UnrealProjectName):
        self.client = client
        self.DataStoragePath = DataStoragePath
        self.size = None
        self.UnrealProjectName = UnrealProjectName
        self.HighResFactor = HighResFactor
        self.HighResPath = f'../../../PackagedEnvironment/{UnrealProjectName}/Demo/Saved/Screenshots/LinuxNoEditor/'
        os.makedirs(self.HighResPath, exist_ok=True)
        self.lowResPath = f'../../../PackagedEnvironment/{UnrealProjectName}/Demo/Binaries/Linux/cache.png'

    def setres(self, w, h):
        print(f'    Windows will be set to {w}x{h}')
        _ = self.client.request(f"vrun r.setres {w}x{h}w")
        self.size = (w, h)

    def isconnected(self):
        return self.client.isconnected()
    
    def QuitGame(self):
        self.client.request('vrun quit')

    def disconnect(self):
        self.client.disconnect()

    def connect(self):
        self.client.connect()

    def getCameraRotation(self, Index=0):
        r = self.client.request(f'vget /camera/{Index}/rotation').split()
        pitch = float(r[0])
        yaw = float(r[1])
        roll = float(r[2])
        return pitch, yaw, roll

    def setCameraRotation(self, pitch, yaw, roll, Index=0):
        _ = self.client.request(f'vset /camera/{Index}/rotation {pitch} {yaw} {roll}')

    def getCameraLocation(self, Index=0):
        r = self.client.request(f'vget /camera/{Index}/location').split()
        x = float(r[0])
        y = float(r[1])
        z = float(r[2])
        return x, y, z

    def setCameraLocation(self, x, y, z, Index=0):
        _ = self.client.request(f'vset /camera/{Index}/location {x} {y} {z}')

    def stepForward(self, walker_type):
        _ = self.client.request(f'vget /object/walk {walker_type}')
        
    def randomizeEnv(self, env_type='All'):
        """
        env_type = 'All', 'light_int', 'light_dir', 'light-color', 'fog'
        """
        _ = self.client.request(f'vget /object/env {env_type}')

    def getNormal(self):
        # output: h x w x 4
        # return with the set size, hHxW3
        imgBinary = self.client.request('vget /camera/0/normal png')
        return WrappedClient.readPNGFromBinary(imgBinary)[:, :, :3]
    def getDepth(self):
        # output: h x w x 4
        # return with the set size, hHxW3
        imgBinary = self.client.request('vget /camera/0/depth npy')
        return WrappedClient.readNPYFromBinary(imgBinary)
    def getColor(self, return_path=False):
        """
        get the lighting map
        return: h x w x 3
        """
        _ = self.client.request(f'vget /camera/0/lit ./cache.png')
        if return_path:
            returns = cv.imread(self.lowResPath), self.lowResPath
        else:
            returns = cv.imread(self.lowResPath)
        return returns

    @staticmethod
    def readPNGFromBinary(binary):
        return np.asarray(Image.open(BytesIO(binary)))
    @staticmethod
    def readNPYFromBinary(binary):
        return np.load(BytesIO(binary))
    # general request
    def request(self, cmd):
        return self.client.request(cmd)
        
    def GetAllTextLocation(self, num, savepath):
        """
        :param textID:
        :return: UL, UR, BR, BL (4, 1, 2)
        """
        _ = self.client.request(f'vget /object/GetAdjusted {num} {savepath}')

    def  LoadTextAttr(self, path):
        _ = self.client.request(f'vget /object/LoadTextAttr {path}')
    def GetWorldCoordinate(self, text_id, path):
        _ = self.client.request(f'vget /object/stickertext/location {text_id} {path}')
    def GetStickerTextCoord(self, idx):
        _ = self.client.request(f'vget /ojbect/stickertext/getlocation {idx}')
    def LoadTextImages(self, path):
        _ = self.client.request(f'vget /object/LoadTextImages {path}')
    
    def SaveImg(self, path):
        _ = self.getCameraLocation()
        if self.HighResFactor > 1.0:
            self.SaveHighRes(path)
        else:
            path = path.replace('jpg', 'png')
            _ = self.client.request(f'vget /screenshot {path}')
            img = cv.imread(path)
            path = path.replace('png', 'jpg')
            cv.imwrite(path, img)
            path = path.replace('jpg', 'png')
            os.system(f'rm {path}')
        # _ = self.getCameraLocation()
    
    def SaveHighRes(self, path):
        _ = self.client.request(f'vrun HighResShot 1.5')
        ID = path.split('/')[-1].split('.')[0]
        t = 0.5
        while True:
            files = list(filter(lambda x:x.find(ID+'.') >= 0, os.listdir(self.HighResPath)))
            if len(files) > 0:
                file = os.path.join(self.HighResPath, files[0])
                break
            else:
                time.sleep(t)
                t *= 2
            if t > 10:
                print("Error in saving!")
                self.SaveImg(path)
                break
        img = cv.imread(file)
        img = cv.resize(img, self.size)
        cv.imwrite(path, img)
    
