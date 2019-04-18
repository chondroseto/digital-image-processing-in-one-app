import sys
import cv2
import math
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtWidgets import QFileDialog,QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
import numpy as np
from matplotlib import pyplot as plt
from konvolusi import convolve as conv
import FindingGradien as fg
import MaxSuppresion as locmax
import HysterisisThres as ht

class ImageProc(QMainWindow):
    def __init__(self):

        super(ImageProc,self).__init__()
        loadUi('showgui.ui',self)
        self.image=None
        newImage=None
        self.loadButton.clicked.connect(self.loadClicked)
        self.saveButton.clicked.connect(self.saveClicked)
        self.action_Load_Image.triggered.connect(self.loadClicked)
        self.action_Save_Image.triggered.connect(self.saveClicked)
        self.actionGrayscale.triggered.connect(self.grayClicked)
        self.actionBrightness.triggered.connect(self.brightClicked)
        self.actionSimple_Contrast.triggered.connect(self.contrastClicked)
        self.actionAuto_Contrast_2.triggered.connect(self.autocontrastClicked)
        self.actionNegative_Image.triggered.connect(self.negatifClicked)
        self.actionGrayscale_Histogram.triggered.connect(self.GrayHistogramClicked)
        self.actionRGB_Histogram.triggered.connect(self.RGBHistogramClicked)
        self.actionBiner_Image.triggered.connect(self.BinerClicked)
        self.actionPerataan_Histogram.triggered.connect(self.EqualHistogramClicked)
        self.action_min_45_Derajat.triggered.connect(self.RotasiMin45Clicked)
        self.action_plus_45Derajat.triggered.connect(self.RotasiPlus45Clicked)
        self.action_Min_90_Derajat.triggered.connect(self.RotasiMin90Clicked)
        self.action_plus_90_Derajat .triggered.connect(self.RotasiPlus90Clicked)
        self.action180_Derajat.triggered.connect(self.Rotasi180Clicked)
        self.actionTranslasi.triggered.connect(self.TranslasiClicked)
        self.actionTranspose.triggered.connect(self.TransposeClicked)
        self.actionLinear_Interpolation.triggered.connect(self.Linear_InterpolationClicked)
        self.actionCubic_Interppolation.triggered.connect(self.Cubic_InterppolationClicked)
        self.actionSkewed_Size.triggered.connect(self.skewed_SizeClicked)
        self.actionCroping.triggered.connect(self.CropingClicked)
        self.actionPenambahan_Citra.triggered.connect(self.aritmatika_CitraClicked)
        self.actionLogika_Not.triggered.connect(self.Logika_NotClicked)
        self.actionLogika_AND.triggered.connect(self.Logika_ANDClicked)
        self.actionSmoothing_Image.triggered.connect(self.SmoothClicked)
        self.actionSharpening_Image.triggered.connect(self.SharpClicked)
        self.actionMean.triggered.connect(self.MeanClicked)
        self.actionMedian.triggered.connect(self.MedianClicked)
        self.actionMax_Filter.triggered.connect(self.MaxClicked)
        self.actionMin_Filter.triggered.connect(self.MinClicked)
        self.actionSobel.triggered.connect(self.SobelClicked)
        self.actionPrewitt.triggered.connect(self.PrewittClicked)
        self.actionCanny.triggered.connect(self.CannyClicked)


    @pyqtSlot()
    def loadClicked(self):
        flname,filter=QFileDialog.getOpenFileName(self,'Open File','D:\\Programming\\Python',"Image Files (*.jpg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    @pyqtSlot()
    def saveClicked(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'D:\\', "Image Files (*.jpg)")
        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Error')

    @pyqtSlot()
    def grayClicked(self):

        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j]= np.clip(0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image=gray
        self.displayImage(2)

    @pyqtSlot()
    def brightClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        brightness = 50
        h, w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = a + brightness
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b

                img.itemset((i, j), b)
        self.image = img
        self.displayImage(2)

    @pyqtSlot()  # contrast
    def contrastClicked(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        width = gray.shape[1]

        contrast = 1.6

        for i in np.arange(height):
            for j in np.arange(width):
                a = gray.item(i, j)
                b = math.ceil(a * contrast)
                if b > 255:
                    b = 255
                gray.itemset((i, j), b)
        self.image = gray
        self.displayImage(2)

    @pyqtSlot()
    def autocontrastClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        min = 255
        max = 0

        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                if a > max:
                    max = a
                if a < min:
                    min = a

        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = float(a - min) / (max - min) * 255
                img.itemset((i, j), b)

        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def BinerClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thres=100
        h, w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a=img.item(i,j)
                if a > thres:
                    a = 1
                else:
                    a = 0
                img.itemset((i,j),a)
                print(img)

        self.image = img
        self.displayImage(2)



    @pyqtSlot()
    def negatifClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        max_intensity = 255
        for i in range(h):
            for j in range(w):
                a = img.item(i, j)
                b = max_intensity - a
                img.itemset((i, j), b)
        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def GrayHistogramClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = img
        self.displayImage(2)
        plt.hist(img.ravel(), 255, [0, 255])
        plt.show()

    @pyqtSlot()
    def RGBHistogramClicked(self):
        color = ('b', 'g', 'r')
        for i,col in enumerate(color):
            histo=cv2.calcHist([self.image],[i],None,[256],[0,256])
            plt.plot(histo,color=col)
            plt.xlim([0,256])
        plt.show()

    @pyqtSlot()
    def EqualHistogramClicked(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

#-------------------------------------------------Operation Geometry----------------------------------------------------------------------#

    @pyqtSlot()
    def TranslasiClicked(self):
        h,w=self.image.shape[:2]
        print(h)
        print(w)
        quarter_h,quarter_w=h/4,w/4
        print(quarter_h)
        print(quarter_w)
        T=np.float32([[1,0,quarter_w],[0,1,quarter_h]])
        print(T)
        img=cv2.warpAffine(self.image,T,(w,h))

        # sx=45
        # sy=-35
        # img=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        # h,w=img.shape[:2]
        # for y in np.arange(1,h):
        #     for x in np.arange(1,w):
        #         xlama=x+sx
        #         ylama=y+sy
        #         img[ylama,xlama]

        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def RotasiMin45Clicked(self):
        self.rotasi(45)
        self.displayImage(2)

    @pyqtSlot()
    def RotasiPlus45Clicked(self):
        self.rotasi(-45)
        self.displayImage(2)

    @pyqtSlot()
    def RotasiMin90Clicked(self):
        self.rotasi(-90)
        self.displayImage(2)

    @pyqtSlot()
    def RotasiPlus90Clicked(self):
        self.rotasi(90)
        self.displayImage(2)

    @pyqtSlot()
    def Rotasi180Clicked(self):
        self.rotasi(180)
        self.displayImage(2)

    def rotasi(self,degree):
        h, w = self.image.shape[:2]

        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.image, rotationMatrix, (h, w))
        self.image=rot_image

    @pyqtSlot()
    def TransposeClicked(self):
        trans_img=cv2.transpose(self.image)
        self.image=trans_img
        self.displayImage(2)

    @pyqtSlot()
    def Linear_InterpolationClicked(self):
        #make size 3/4 original image size
        cv2.imshow('Original',self.image)
        resize_img=cv2.resize(self.image,None,fx=0.50, fy=0.50)
        self.image=resize_img
        cv2.imshow('',self.image)
        #self.displayImage(2)

    @pyqtSlot()
    def Cubic_InterppolationClicked(self):
        #double size of original image size/zooming(scaling up)
        cv2.imshow('Original', self.image)
        resize_img=cv2.resize(self.image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        self.image = resize_img
        cv2.imshow('',self.image)
        #self.displayImage(2)

    @pyqtSlot()
    def skewed_SizeClicked(self):
        #resize image based on exacat dimension
        cv2.imshow('Original', self.image)
        resize_img=cv2.resize(self.image,(900,400),interpolation=cv2.INTER_AREA)
        self.image=resize_img
        cv2.imshow('',self.image)
        #self.displayImage(2)

    @pyqtSlot()
    def CropingClicked(self):
        h,w=self.image.shape[:2]
        #get the strating point of pixel coord(top left)
        start_row, start_col=int(h*.1),int(w*.1)
        #get the ending point coord (botoom right)
        end_row, end_col=int(h*.5),int(w*.5)
        crop=self.image[0:1000,0:500]
        cv2.imshow('Original',self.image)
        cv2.imshow('Crop Image',crop)

#-----------------------------------operasi aritmatika---------------------------------------------------
    @pyqtSlot()
    def aritmatika_CitraClicked(self):
        img1 = cv2.imread('img1.jpg', 0)
        img2 = cv2.imread('img2.jpg', 0)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        add_img = img1 + img2
        subtract = img1 - img2
        subtract2=img2-img1
        mul = img1 * img2
        div = img1 / img2

        # titles=['Image 1','Image 2','Add','Subtract (1-2)','Subtract(2-1)','Multiply','Divided']
        # images=[img1,img2,add_img,subtract,subtract2,mul,div]
        # print()
        #
        # for i in range(8):
        #     plt.subplot(1,8,i+1)
        #     plt.imshow(images[i])
        #     plt.title(titles[i])
        #     plt.xticks([])
        #     plt.yticks([])
        # plt.show()
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('Add', add_img)
        cv2.imshow('Subtraction', subtract)
        cv2.imshow('Multiply', mul)
        cv2.imshow('Divide', div)

    @pyqtSlot()
    def Logika_NotClicked(self):
        #
        img=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        #cv2.bitwise_not(img)
        img=cv2.bitwise_not(img)
        self.image=img
        self.displayImage(2)
        #
        # img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # thres = 100
        # h, w = img.shape[:2]
        # for i in np.arange(h):
        #     for j in np.arange(w):
        #         a = img.item(i, j)
        #         if a > thres:
        #             a = 255
        #         elif a < thres:
        #             a = 0
        #         else:
        #             a = a
        #         img=img.itemset((i, j), a)
        #         img2=cv2.bitwise_not(img)
        #
        # self.image=img2
        # self.displayImage(2)


    @pyqtSlot()
    def Logika_ANDClicked(self):
        img1 = cv2.imread('img1.jpg', 1)
        img2 = cv2.imread('img2.jpg', 1)
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_and=cv2.bitwise_and(img1,img2)
        op_or=cv2.bitwise_or(img2,img2)
        op_xor=cv2.bitwise_xor(img1,img2)

        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('And', op_and)
        cv2.imshow('OR', op_or)
        cv2.imshow('XOR', op_xor)

#--------------------------------------------------------OPERASI SPASIAL------------------------------------------------------------------#
    @pyqtSlot()
    def SmoothClicked(self):
        img=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        #h,w=img.shape[:2]
        gauss =(1.0 / 57)* np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 5, 3, 1],
             [2, 5, 9, 5, 2],
             [1, 3, 5, 3, 1],
             [0, 1, 2, 1, 0]])

        img_out=conv(img,gauss)

        #
        # #cv2.imshow('',img_out)
        # self.image=img_out
        # self.displayImage(i)

        plt.imshow(img_out,cmap='gray',interpolation='bicubic')
        plt.xticks([],plt.yticks([]))
        plt.show()

    pyqtSlot()
    def SharpClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        laplace = (1.0 / 16) * np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]])

        img_out = conv(img, laplace)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    pyqtSlot()
    def MeanClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        laplace = (1.0 / 9) * np.array(
            [[1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1]])

        img_out = conv(img, laplace)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    pyqtSlot()
    def MedianClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out=img.copy()
        h,w=img.shape[:2]

        for i in np.arange(3,h-3):
            for j in np.arange(3,w-3):
                neighbors=[]
                for k in np.arange(-3,4):
                    for l in np.arange(-3,4):
                        a=img.item(i+k, j+l)
                        neighbors.append(a)
                neighbors.sort()
                median=neighbors[24]
                b=median
                img_out.itemset((i,j),b)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    @pyqtSlot()
    def MaxClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max=a
                b=max
                img_out.itemset((i, j), b)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    @pyqtSlot()
    def MinClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min=a
                b=min
                img_out.itemset((i, j), b)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()
#--------------------------------------------------------ANALISIS CITRA------------------------------------------------------------------------------#
    @pyqtSlot()
    def SobelClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = conv(img,Sx)/8.0
        img_y = conv(img,Sy)/8.0
        img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))
        img_out = (img_out / np.max(img_out)) * 255

        self.image=img

        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    @pyqtSlot()
    def PrewittClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        Px = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
        Py = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])

        img_x = conv(img, Px) / 6.0
        img_y = conv(img, Py) / 6.0

        img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))
        img_out = (img_out / np.max(img_out)) * 255

        self.image = img
        self.displayImage(2)

        plt.imshow(img_out,cmap = 'gray',interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    @pyqtSlot()
    def CannyClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h,w=img.shape[:2]

        #step 1: reduce noise with Gaussian Filter
        gauss = (1.0 / 57) * np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 5, 3, 1],
             [2, 5, 9, 5, 2],
             [1, 3, 5, 3, 1],
             [0, 1, 2, 1, 0]])

        img_blur = conv(img, gauss)

        #step 2: Finding Gradien
        img_x=conv(img_blur,np.array([[-0.5,0,0.5]]))
        img_y=conv(img_blur,np.array([[-0.5,0,0.5]]))

        E_mag=np.sqrt(np.power(img_x,2)+np.power(img_y,2))
        E_mag=(E_mag/np.max(E_mag))*255

        #step 3: non-maximum suppresion
        t_low=4
        E_nms=np.zeros((h,w))
        for i in np.arange(1,h-1):
            for j in np.arange(1,w-1):
                dx=img_x[i,j]
                dy=img_y[i,j]
                s_theta=fg.FindingGradien(dx,dy)

                if locmax.MaxSuppesion(E_mag,i,j,s_theta,t_low):
                    E_nms[i,j]=E_mag[i,j]

        #step 4: Hysterisis Thresholding
        t_high=15
        E_bin=np.zeros((h,w))
        for i in np.arange(1,h-1):
            for j in np.arange(1,w-1):
                if E_nms[i,j]>=t_high and E_bin[i,j]==0:
                    ht.HysterisisThres(E_nms,E_bin,i,j,t_low)

        self.image=img
        self.displayImage(2)

        plt.imshow(E_bin,cmap='gray', interpolation='bicubic')
        plt.xticks([]),plt.yticks([])
        plt.show()










#--------------------------------------------------------FUNGSI UMUM----------------------------------------------------------------------#

    def loadImage(self,flname):
        self.image=cv2.imread(flname,cv2.IMREAD_COLOR)
        self.displayImage()

    def displayImage(self, windows=1):
        qformat=QImage.Format_Indexed8

        if len(self.image.shape)==3:
            if(self.image.shape[2])==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        img=QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],qformat)

        #BGR>RGB
        img=img.rgbSwapped()
        if windows==1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)
        if windows==2:
            self.hasilLabel.setPixmap(QPixmap.fromImage(img))
            self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.hasilLabel.setScaledContents(True)

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    window=ImageProc()
    window.setWindowTitle('Image Processing')
    window.show()
    sys.exit(app.exec_())
